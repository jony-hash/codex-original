"""Content-bound, research-only measurements for the v2.2 capacity map.

This module intentionally does not import the historical diagnostic/current
runtimes.  The fixed panels are reproduced here so a capacity-map evidence
record always identifies the exact measurements that produced its conclusion.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ...evidence import canonical_sha256, tensor_sha256
from ...student.conditional import camera_code_fraction, log_sinkhorn
from ...student.evaluator import stratified_mnist_indices


MEASUREMENT_CONTRACT = canonical_sha256(
    {
        "kind": "capacity_map_v22_dual_evaluator_measurement",
        "panels": ["axis", "uniform_1234", "uniform_5678", "holdout_9012"],
        "samples_per_panel": 170,
        "reference": "mnist_stratified_validation_5000",
        "exposure_scales": [2.0**(value / 2.0) for value in range(-4, 5)],
    }
)
PANEL_NAMES = ("axis", "uniform_1234", "uniform_5678", "holdout_9012")
EXPOSURE_SCALES = tuple(2.0 ** (value / 2.0) for value in range(-4, 5))


def fixed_panels(*, device: torch.device, code_dim: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return the v2.1 10x17 panels; v2.2 O0 reconciliation is 8D only."""
    if code_dim != 8:
        raise ValueError("v2.2 full-panel reconciliation is pre-registered for E2's 8D code only")
    axis = torch.eye(8, device=device)
    axis_codes = torch.cat((torch.zeros(1, 8, device=device), axis, -axis), dim=0)
    labels = torch.arange(10, device=device).repeat_interleave(17)
    def generator_codes(seed: int) -> torch.Tensor:
        return torch.randn((170, 8), generator=torch.Generator(device=device).manual_seed(seed), device=device)
    return {
        "axis": (axis_codes.repeat(10, 1), labels),
        "uniform_1234": (generator_codes(1234), labels),
        "uniform_5678": (generator_codes(5678), labels),
        "holdout_9012": (generator_codes(9012), labels),
    }


def panel_hashes(*, device: torch.device, code_dim: int) -> dict[str, str]:
    return {
        name: canonical_sha256({"codes": tensor_sha256(codes.cpu()), "labels": tensor_sha256(labels.cpu())})
        for name, (codes, labels) in fixed_panels(device=device, code_dim=code_dim).items()
    }


def validation_reference(images: torch.Tensor, labels: torch.Tensor, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the same immutable 5,000-image stratified validation split as v2.1."""
    _, validation_indices = stratified_mnist_indices(labels)
    return images.index_select(0, validation_indices).to(device), labels.index_select(0, validation_indices).to(device)


def evaluator_features(model: nn.Module, images: torch.Tensor, *, batch_size: int = 250) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate reference data in bounded batches for the 6 GB CUDA contract."""
    features, logits = [], []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            feature, output = model(images[start : start + batch_size])
            features.append(feature)
            logits.append(output)
    return torch.cat(features), torch.cat(logits)


def _recall(prediction: torch.Tensor, labels: torch.Tensor) -> list[float]:
    return [float((prediction[labels == label] == label).float().mean().item()) for label in range(10)]


def _bootstrap_ci(prediction: torch.Tensor, labels: torch.Tensor, *, seed: int) -> list[float]:
    correct = (prediction == labels).float().cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = []
    for _ in range(1_000):
        index = torch.randint(correct.numel(), (correct.numel(),), generator=generator)
        samples.append(correct.index_select(0, index).mean())
    values = torch.stack(samples)
    return [float(torch.quantile(values, value).item()) for value in (0.025, 0.975)]


def _rank_summary(rows: torch.Tensor) -> dict[str, float | int]:
    singular = torch.linalg.svdvals(rows.float())
    if singular.numel() < 8:
        return {"rank": int(singular.numel()), "sigma8_over_sigma1": 0.0}
    return {
        "rank": int((singular >= singular[0].clamp_min(1.0e-12) * 1.0e-3).sum().item()),
        "sigma8_over_sigma1": float((singular[7] / singular[0].clamp_min(1.0e-12)).item()),
    }


def _distribution(features: torch.Tensor, reference: torch.Tensor, labels: torch.Tensor, reference_labels: torch.Tensor) -> dict[str, object]:
    ratios, sinkhorn, precision, recall, nearest_ratio = [], [], [], [], []
    for label in range(10):
        generated = F.normalize(features[labels == label], dim=1)
        real = F.normalize(reference[reference_labels == label], dim=1)
        selected_real = real[: generated.shape[0]]
        generated_distance = torch.pdist(generated).median()
        real_distance = torch.pdist(selected_real).median().clamp_min(1.0e-8)
        ratios.append(float((generated_distance / real_distance).item()))
        cost = torch.cdist(generated, selected_real).square()
        sinkhorn.append(float(log_sinkhorn(cost, epsilon=0.05, iterations=20).item()))
        real_cost = torch.cdist(selected_real, selected_real)
        radius = torch.quantile(real_cost[~torch.eye(real_cost.shape[0], dtype=torch.bool, device=real_cost.device)], 0.20)
        generated_nearest = cost.sqrt().min(dim=1).values
        real_nearest = cost.sqrt().min(dim=0).values
        precision.append(float((generated_nearest <= radius).float().mean().item()))
        recall.append(float((real_nearest <= radius).float().mean().item()))
        real_nearest_baseline = real_cost.masked_fill(torch.eye(real_cost.shape[0], dtype=torch.bool, device=real_cost.device), float("inf")).sqrt().min(dim=1).values.median()
        nearest_ratio.append(float((generated_nearest.median() / real_nearest_baseline.clamp_min(1.0e-8)).item()))
    return {
        "per_class_distance_ratio": ratios,
        "sinkhorn_coverage": sinkhorn,
        "precision": precision,
        "recall": recall,
        "memorization_nearest_distance_ratio": nearest_ratio,
    }


def _anti_template(images: torch.Tensor, features: torch.Tensor, codes: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    values = {}
    for label in range(10):
        selected = labels == label
        rows, code = images[selected].flatten(1), codes[selected]
        centered = rows - rows.mean(0, keepdim=True)
        output_distance, code_distance = torch.pdist(rows), torch.pdist(code)
        correlation = 0.0
        if output_distance.std() > 0 and code_distance.std() > 0:
            correlation = float(torch.corrcoef(torch.stack((output_distance, code_distance)))[0, 1].item())
        values[str(label)] = {
            "feature": _rank_summary(features[selected] - features[selected].mean(0, keepdim=True)),
            "image": _rank_summary(centered),
            "unique_quantized": len({tensor_sha256(((row * 255).round() / 255).cpu()) for row in images[selected]}),
            "code_output_correlation": correlation,
        }
    return values


def _command_jacobian(model: nn.Module, *, device: torch.device) -> dict[str, object]:
    """Forward-mode directional Jacobians avoid materialising a 40k x 40k graph."""
    result: dict[str, object] = {}
    for label in range(10):
        code = torch.zeros(8, device=device, requires_grad=True)
        target = torch.tensor([label], device=device)
        columns = []
        for direction in torch.eye(8, device=device):
            _, derivative = torch.autograd.functional.jvp(
                lambda value: model(value.unsqueeze(0), target).flatten(), code, direction, create_graph=False
            )
            columns.append(derivative)
        singular = torch.linalg.svdvals(torch.stack(columns))
        result[str(label)] = {
            "effective_rank": int((singular >= singular[0].clamp_min(1.0e-12) * 1.0e-3).sum().item()),
            "singular_values": [float(value) for value in singular.cpu()],
        }
    return result


def _curvature(render: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], *, device: torch.device) -> dict[str, float]:
    labels = torch.arange(10, device=device)
    zero = torch.zeros((10, 8), device=device)
    base = render(zero, labels)
    values = []
    for direction in torch.eye(8, device=device):
        plus, minus = zero + direction, zero - direction
        second = render(plus, labels) + render(minus, labels) - 2 * base
        values.append(float(second.square().mean().sqrt().item()))
    return {"axis_rms": values, "minimum_axis_rms": min(values)}


def measure_candidate(
    *,
    model: nn.Module,
    render: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    real_images: torch.Tensor,
    real_labels: torch.Tensor,
    primary: nn.Module,
    smooth: nn.Module,
    device: torch.device,
    shape: nn.Module | None = None,
    output_domain: str,
) -> dict[str, object]:
    """Measure all panels without optimizer creation or generated-data selection."""
    reference, reference_labels = validation_reference(real_images, real_labels, device=device)
    primary_reference, _ = evaluator_features(primary, reference)
    smooth_reference, _ = evaluator_features(smooth, reference)
    shape_reference = evaluator_features(shape, reference)[0] if shape is not None else None
    panels, details = fixed_panels(device=device, code_dim=8), {}
    for number, (name, (codes, labels)) in enumerate(panels.items()):
        with torch.no_grad():
            images = render(codes, labels).clamp(0, 1)
            primary_features, primary_logits = primary(images)
            smooth_features, smooth_logits = smooth(images)
            shape_features, shape_logits = shape(images) if shape is not None else (None, None)
        primary_prediction, smooth_prediction = primary_logits.argmax(1), smooth_logits.argmax(1)
        row: dict[str, object] = {
            "primary": {"accuracy": float((primary_prediction == labels).float().mean().item()), "recall": _recall(primary_prediction, labels), "bootstrap_95_ci": _bootstrap_ci(primary_prediction, labels, seed=10_000 + number)},
            "smooth": {"accuracy": float((smooth_prediction == labels).float().mean().item()), "recall": _recall(smooth_prediction, labels), "bootstrap_95_ci": _bootstrap_ci(smooth_prediction, labels, seed=20_000 + number)},
            "agreement": float((primary_prediction == smooth_prediction).float().mean().item()),
            "primary_distribution": _distribution(primary_features, primary_reference, labels, reference_labels),
            "smooth_distribution": _distribution(smooth_features, smooth_reference, labels, reference_labels),
            "anti_template": {"primary": _anti_template(images, primary_features, codes, labels), "smooth": _anti_template(images, smooth_features, codes, labels)},
            "digital_output_code_fraction": float(camera_code_fraction(images, labels).item()),
            "saturation": float((images >= 1).float().mean().item()),
        }
        if shape is not None and shape_features is not None and shape_logits is not None and shape_reference is not None:
            prediction = shape_logits.argmax(1)
            row["shape"] = {"accuracy": float((prediction == labels).float().mean().item()), "recall": _recall(prediction, labels), "bootstrap_95_ci": _bootstrap_ci(prediction, labels, seed=30_000 + number), "distribution": _distribution(shape_features, shape_reference, labels, reference_labels), "agreement_primary": float((prediction == primary_prediction).float().mean().item()), "agreement_smooth": float((prediction == smooth_prediction).float().mean().item())}
        details[name] = row
    return {
        "measurement_contract": MEASUREMENT_CONTRACT,
        "panel_hashes": panel_hashes(device=device, code_dim=8),
        "panels": details,
        "command_jacobian": _command_jacobian(model, device=device),
        "command_curvature": _curvature(lambda codes, labels: model(codes, labels), device=device),
        "output_curvature": _curvature(render, device=device),
        "physical": {"raw_cmos": "not_applicable" if output_domain == "electronic_oracle" else "reported_by_operator", "crop_retention": "not_applicable" if output_domain == "electronic_oracle" else "reported_by_operator"},
    }


def exposure_sensitivity(
    *, render: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], primary: nn.Module, smooth: nn.Module, device: torch.device
) -> dict[str, object]:
    """Report an immutable global scale sweep; it does not select an exposure."""
    codes, labels = fixed_panels(device=device, code_dim=8)["holdout_9012"]
    with torch.no_grad():
        raw = render(codes, labels)
        report = {}
        for scale in EXPOSURE_SCALES:
            images = (raw * scale).clamp(0, 1)
            _, primary_logits = primary(images)
            _, smooth_logits = smooth(images)
            primary_prediction, smooth_prediction = primary_logits.argmax(1), smooth_logits.argmax(1)
            report[str(scale)] = {"primary_accuracy": float((primary_prediction == labels).float().mean().item()), "smooth_accuracy": float((smooth_prediction == labels).float().mean().item()), "agreement": float((primary_prediction == smooth_prediction).float().mean().item()), "saturation": float((images >= 1).float().mean().item())}
    return {"selection_permitted": False, "scales": list(EXPOSURE_SCALES), "holdout_9012": report}
