"""Post-training acceptance metrics for the conditional optical dataset."""
from __future__ import annotations

from pathlib import Path
import torch

from ..evidence import canonical_sha256, file_sha256, safe_torch_load
from .conditional import conditional_sinkhorn_loss


def per_label_feature_diversity_ratio(generated: torch.Tensor, real: torch.Tensor, labels: torch.Tensor) -> tuple[float, ...]:
    values = []
    for label in range(10):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() < 2:
            raise ValueError("diversity acceptance requires at least two samples per label")
        generated_distance = torch.pdist(generated.index_select(0, indices)).median()
        real_distance = torch.pdist(real.index_select(0, indices)).median().clamp_min(1.0e-12)
        values.append(float((generated_distance / real_distance).item()))
    return tuple(values)


def conditional_coverage_ratio(
    generated_features: torch.Tensor,
    generated_view: torch.Tensor,
    real_features: torch.Tensor,
    real_view: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float,
    iterations: int,
    seed: int = 1234,
) -> float:
    """Compare generated coverage with two disjoint real subsets.

    The old implementation compared one real set with a permutation of
    itself.  That is an OT identity and is not a coverage baseline.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    generated_rows, left_rows, right_rows = [], [], []
    count: int | None = None
    for label in range(10):
        generated_indices = torch.nonzero(labels == label, as_tuple=False).flatten().cpu()
        real_indices = torch.nonzero(labels == label, as_tuple=False).flatten().cpu()
        if generated_indices.numel() < 2 or real_indices.numel() < 4:
            raise ValueError("coverage requires at least two generated and four real samples per label")
        shuffled = real_indices.index_select(0, torch.randperm(real_indices.numel(), generator=generator))
        half = shuffled.numel() // 2
        current = min(generated_indices.numel(), half)
        count = current if count is None else min(count, current)
        generated_rows.append(generated_indices)
        left_rows.append(shuffled[:current])
        right_rows.append(shuffled[current : current * 2])
    assert count is not None
    generated_selected = torch.cat([rows[:count] for rows in generated_rows])
    left_selected = torch.cat([rows[:count] for rows in left_rows])
    right_selected = torch.cat([rows[:count] for rows in right_rows])
    local_labels = torch.arange(10, device=labels.device).repeat_interleave(count)
    observed = conditional_sinkhorn_loss(
        generated_features.index_select(0, generated_selected.to(labels.device)),
        generated_view.index_select(0, generated_selected.to(labels.device)),
        real_features.index_select(0, right_selected.to(labels.device)),
        real_view.index_select(0, right_selected.to(labels.device)),
        local_labels,
        epsilon=epsilon,
        iterations=iterations,
    )
    baseline = conditional_sinkhorn_loss(
        real_features.index_select(0, left_selected.to(labels.device)),
        real_view.index_select(0, left_selected.to(labels.device)),
        real_features.index_select(0, right_selected.to(labels.device)),
        real_view.index_select(0, right_selected.to(labels.device)),
        local_labels,
        epsilon=epsilon,
        iterations=iterations,
    )
    return float((observed / baseline.clamp_min(1.0e-12)).item())


def chunked_nearest_distances(query: torch.Tensor, reference: torch.Tensor, *, chunk_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest squared-L2 distance/index without constructing N×M globally."""
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1] or reference.numel() == 0:
        raise ValueError("nearest-neighbor inputs must be non-empty [N, D] tensors")
    distances, indices = [], []
    for start in range(0, query.shape[0], chunk_size):
        cost = torch.cdist(query[start : start + chunk_size], reference).square()
        value, index = cost.min(dim=1)
        distances.append(value.cpu())
        indices.append(index.cpu())
    return torch.cat(distances), torch.cat(indices)


def memorization_statistics(generated_features: torch.Tensor, generated_labels: torch.Tensor, real_features: torch.Tensor, real_labels: torch.Tensor, *, seed: int = 1234) -> dict[str, object]:
    """Per-class 1% real-real threshold and generated nearest-neighbor audit."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    per_class = []
    all_flags = []
    for label in range(10):
        real_indices = torch.nonzero(real_labels.cpu() == label, as_tuple=False).flatten()
        generated_indices = torch.nonzero(generated_labels.cpu() == label, as_tuple=False).flatten()
        if real_indices.numel() < 2 or generated_indices.numel() == 0:
            raise ValueError("memorization audit requires real and generated examples for every label")
        shuffled = real_indices.index_select(0, torch.randperm(real_indices.numel(), generator=generator))
        midpoint = shuffled.numel() // 2
        left, right = shuffled[:midpoint], shuffled[midpoint:]
        left_features, right_features = real_features.index_select(0, left), real_features.index_select(0, right)
        left_distances, _ = chunked_nearest_distances(left_features, right_features)
        right_distances, _ = chunked_nearest_distances(right_features, left_features)
        threshold = float(torch.quantile(torch.cat((left_distances, right_distances)), 0.01).item())
        nearest, nearest_local = chunked_nearest_distances(generated_features.index_select(0, generated_indices), real_features.index_select(0, real_indices))
        flags = nearest <= threshold
        all_flags.append(flags)
        per_class.append({"label": label, "threshold": threshold, "anomalous_rate": float(flags.to(torch.float32).mean().item()), "nearest_indices": real_indices.index_select(0, nearest_local).tolist(), "nearest_distances": nearest.tolist()})
    overall = float(torch.cat(all_flags).to(torch.float32).mean().item())
    return {"seed": seed, "overall_anomalous_rate": overall, "per_class": per_class}


def load_candidate_dataset(path: str | Path) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    root = Path(path)
    manifest_path = root / "manifest.json"
    import json

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = dict(payload)
    manifest_hash = expected.pop("manifest_hash", None)
    if payload.get("status") != "candidate" or manifest_hash != canonical_sha256(expected):
        raise ValueError("candidate dataset manifest is invalid or not a candidate")
    images, labels = [], []
    for shard in payload.get("shards", []):
        shard_path = root / str(shard["path"])
        if not shard_path.is_file() or file_sha256(shard_path) != shard["sha256"]:
            raise ValueError("candidate dataset shard hash changed")
        value = safe_torch_load(shard_path, expected_sha256=str(shard["sha256"]))
        images.append(value["mnist_view_uint8"].to(torch.float32).div_(255.0))
        labels.append(value["labels"].to(torch.long))
    if not images:
        raise ValueError("candidate dataset contains no shards")
    return payload, torch.cat(images), torch.cat(labels)


def paired_bootstrap_lower_bound(differences: torch.Tensor, *, seed: int = 1234, resamples: int = 10000) -> float:
    if differences.ndim != 1 or differences.numel() == 0:
        raise ValueError("bootstrap requires one difference per test example")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = torch.randint(0, differences.numel(), (resamples, differences.numel()), generator=generator)
    means = differences.cpu().index_select(0, samples.flatten()).reshape(resamples, -1).mean(dim=1)
    return float(torch.quantile(means, 0.05).item())
