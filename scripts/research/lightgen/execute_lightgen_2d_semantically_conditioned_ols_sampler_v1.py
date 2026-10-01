"""Execute the frozen semantic-conditioned OLS sampler comparison.

This executor fits only the preregistered S0--S3 statistical samplers in OLS
space.  Candidate selection is made from held-out TRAIN latent diagnostics;
the frozen step3000 Generator is loaded and executed only after the winner is
frozen, for one winner canary and one S0 empirical-support control.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import EPS_S, OLS_PORT_COUNT, max_q_gauge, statistical_variable, wrap_phase
from ldm_project.v2.evidence import file_sha256, safe_torch_load, tensor_sha256
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1 as canary
from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as robust
from scripts.research.lightgen import execute_lightgen_2d_ols_to_cmos_semantic_information_localization_v1 as localization
from scripts.research.lightgen import execute_lightgen_2d_post3000_generation_readiness_scaling_v1 as readiness
from scripts.research.lightgen import execute_lightgen_2d_real_vs_sampled_ols_semantic_manifold_gap_v1 as gap


TASK_ID = "EXECUTE_SEMANTICALLY_CONDITIONED_OLS_SAMPLER_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_SEMANTICALLY_CONDITIONED_OLS_SAMPLER_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "SEMANTICALLY_CONDITIONED_OLS_SAMPLER_V1_RESOLVED"
INPUT_GATE = "SEMANTICALLY_CONDITIONED_OLS_SAMPLER_CONTRACT_DEFINED"
SUPPORTED_BOTTLENECK = "SAMPLED_OLS_SEMANTIC_STRUCTURE_NOT_PRESERVED"
EXPECTED_BRANCH = readiness.EXPECTED_BRANCH
EXPECTED_HEAD = readiness.EXPECTED_HEAD

TRAIN_OLS = Path("outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01/final_train_ols.pt")
TRAIN_OLS_SHA256 = "2d1264219faa0964467d40f2e4882ec3767d3a1aafd30e143bf406aa39be6dcc"
DEV_OLS = Path("outputs/20260918_frozen_dev_ols_cmos_v1/runs/attempt_01/dev_complex_ols.pt")
DEV_OLS_SHA256 = "5a8efe07c6ba52168f415e56fccdd68c2552b6dd5f4f975dabc96d2215fbb622"
PROBE_WEIGHTS = Path("outputs/20260919_ols_to_cmos_semantic_information_localization_v1/runs/attempt_01/probe_weights.pt")
PROBE_WEIGHTS_SHA256 = "b76e1d22420bd48e620a3d293ca15e8d6e04ab192df5cc5bee8c1cdb5310b383"
P2Q2_BANK = Path("outputs/20260918_chen_bogt_phase_completion_repair_v1/runs/attempt_01/selected_phase_completion_latent_bank.pt")
P2Q2_BANK_SHA256 = "2f374dfb16b8b95488ee5bf1b0ee776aaee6d2d37128ab6c4dc606d7cc308db2"
GAP_REPORT = Path("outputs/20260920_real_vs_sampled_ols_semantic_manifold_gap_v1/runs/attempt_08/report.json")
GAP_REPORT_SHA256 = "3793cfbbbdf7ed688795ec5d07af40ac3d70777f72260520661b77299de69ef7"
DEFAULT_OUTPUT_DIR = Path("outputs/20260920_semantically_conditioned_ols_sampler_v1/runs/attempt_01")

CLASS_COUNT = 10
TRAIN_COUNT = 55_000
DEV_COUNT = 5_000
BANK_PER_CLASS = 200
BANK_COUNT = CLASS_COUNT * BANK_PER_CLASS
SAMPLER_VALIDATION_PER_CLASS = 500
VALIDATION_HALF_PER_CLASS = 250
S1_NEIGHBOR_COUNT = 16
PCA_VARIANCE_FRACTION = 0.95
PCA_RANK_CAP = 32
RESIDUAL_VARIANCE_FLOOR = 1.0e-8
SPLIT_SEED = 2_026_092_001
BANK_SEED = 2_026_092_002
BOOTSTRAP_COUNT = 999
BOOTSTRAP_SEED = 271_828
DISTANCE_QUERY_CHUNK = 128
CANARY_BATCH_SIZE = 10
METRIC_PER_CLASS = 20
LATENT_TARGET = 0.90
GENERATOR_TARGET = 0.90
OLD_Q2_PREDCLASS_SUPPORT = 0.245

PROBE_MEAN_SHA256 = "5c71f3111e653cd6643c3c48246f84b7104cf4d7c8d847011b0ae4197cde22c8"
PROBE_STD_SHA256 = "981cfaf7d151b3305f6406c75f053199f8a4cfd2a28076aa94fd2c05a9fd7d48"

SIMPLICITY_ORDER = {"S1": 0, "S2": 1, "S3": 2}


class SamplerExecutionError(RuntimeError):
    """Raised when a frozen authority or scientific boundary cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SamplerExecutionError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _float_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1)]


def _int_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.detach().cpu().reshape(-1)]


def _json_float(value: object) -> float:
    result = float(value)
    _require(math.isfinite(result), "nonfinite JSON float")
    return result


def _summary(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().cpu().to(torch.float64).reshape(-1)
    _require(flat.numel() > 0, "summary input is empty")
    _finite(flat, "summary input")
    quantiles = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float64))
    return {
        "count": int(flat.numel()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std_population": float(flat.std(unbiased=False)),
        "q50": float(quantiles[0]),
        "q90": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "q99": float(quantiles[3]),
    }


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"{role} is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _write_json(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    robust._write_json(path, payload, kind=kind)


def _write_torch(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    robust._write_torch(path, payload, kind=kind)


def _ols_features(z: torch.Tensor) -> torch.Tensor:
    _require(z.ndim == 2 and z.shape[1] == OLS_PORT_COUNT and torch.is_complex(z), "OLS must be complex [N,100]")
    value = torch.cat((z.real, z.imag), dim=1).to(torch.float64).contiguous()
    _finite(value, "OLS Cartesian features")
    return value


def _fit_standardizer(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = value.to(torch.float64).contiguous()
    mean = value.mean(dim=0)
    raw_std = value.std(dim=0, unbiased=False)
    safe_std = torch.where(raw_std > 0.0, raw_std, torch.ones_like(raw_std))
    _finite(mean, "fit standardization mean")
    _finite(safe_std, "fit standardization std")
    return ((value - mean) / safe_std).contiguous(), mean.contiguous(), safe_std.contiguous()


def _apply_standardizer(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    result = (value.to(torch.float64) - mean) / std
    _finite(result, "standardized query")
    return result.contiguous()


def _complex_from_cartesian(value: torch.Tensor) -> torch.Tensor:
    _require(value.ndim == 2 and value.shape[1] == 2 * OLS_PORT_COUNT, "Cartesian representation has the wrong shape")
    z = torch.complex(value[:, :OLS_PORT_COUNT], value[:, OLS_PORT_COUNT:]).to(torch.complex64)
    _finite(z.real, "reconstructed OLS real part")
    _finite(z.imag, "reconstructed OLS imaginary part")
    return z.contiguous()


def _confusion(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    labels = labels.to(torch.long).reshape(-1)
    predictions = predictions.to(torch.long).reshape(-1)
    _require(labels.shape == predictions.shape and labels.numel() > 0, "classification inputs differ")
    result = torch.zeros((CLASS_COUNT, CLASS_COUNT), dtype=torch.long)
    for actual, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        _require(0 <= actual < CLASS_COUNT and 0 <= predicted < CLASS_COUNT, "class outside 0..9")
        result[actual, predicted] += 1
    return result


def _classification_report(logits: torch.Tensor, requested: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    logits = logits.detach().cpu().to(torch.float64)
    requested = requested.detach().cpu().to(torch.long)
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    matrix = _confusion(requested, predictions)
    recalls = matrix.diag().to(torch.float64) / matrix.sum(dim=1).to(torch.float64)
    ordered = torch.sort(probabilities, dim=1, descending=True).values
    entropy = -(probabilities * probabilities.clamp_min(1.0e-300).log()).sum(dim=1)
    return (
        {
            "labels_consumed": True,
            "label_source": "requested class y supplied to sample(y, seed_or_index)",
            "rows": int(requested.numel()),
            "accuracy": float((requested == predictions).to(torch.float64).mean()),
            "macro_recall": float(recalls.mean()),
            "per_class_recall": _float_list(recalls),
            "confusion_matrix_requested_rows_predicted_columns": matrix.tolist(),
            "predicted_class_histogram": _int_list(torch.bincount(predictions, minlength=CLASS_COUNT)),
            "requested_class_histogram": _int_list(torch.bincount(requested, minlength=CLASS_COUNT)),
            "top1_confidence": _summary(ordered[:, 0]),
            "top1_top2_logit_margin": _summary(logits.topk(2, dim=1).values[:, 0] - logits.topk(2, dim=1).values[:, 1]),
            "prediction_entropy_nats": _summary(entropy),
        },
        predictions,
        probabilities,
    )


def _probe_logits(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, probe: Mapping[str, Any]) -> torch.Tensor:
    features = _apply_standardizer(_ols_features(z), mean, std)
    weights = _tensor(probe["weights"], "probe weights")
    bias = _tensor(probe["bias"], "probe bias")
    logits = features @ weights.T + bias
    _finite(logits, "probe logits")
    return logits


def _nearest_distance(query: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    query = query.to(torch.float64).cpu().contiguous()
    reference = reference.to(torch.float64).cpu().contiguous()
    _require(query.ndim == 2 and reference.ndim == 2 and query.shape[1] == reference.shape[1], "distance dimensions differ")
    distances: list[torch.Tensor] = []
    locations: list[torch.Tensor] = []
    for start in range(0, query.shape[0], DISTANCE_QUERY_CHUNK):
        stop = min(start + DISTANCE_QUERY_CHUNK, query.shape[0])
        block = torch.cdist(query[start:stop], reference, p=2)
        values, indices = block.min(dim=1)
        distances.append(values)
        locations.append(indices.to(torch.long))
    result_distance = torch.cat(distances)
    result_index = torch.cat(locations)
    _finite(result_distance, "nearest distances")
    return result_distance, result_index


def _entropy_from_histogram(histogram: torch.Tensor) -> tuple[float, float]:
    probabilities = histogram.to(torch.float64)
    probabilities = probabilities / probabilities.sum()
    nonzero = probabilities > 0.0
    entropy = float(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
    return entropy, entropy / math.log(float(CLASS_COUNT))


def _make_sampler_split(train_labels: torch.Tensor, train_indices: torch.Tensor) -> dict[str, Any]:
    _require(train_labels.shape == (TRAIN_COUNT,), "TRAIN label shape changed")
    generator = torch.Generator(device="cpu").manual_seed(SPLIT_SEED)
    fit_positions_by_class: list[torch.Tensor] = []
    validation_positions_by_class: list[torch.Tensor] = []
    va_positions_by_class: list[torch.Tensor] = []
    vb_positions_by_class: list[torch.Tensor] = []
    for digit in range(CLASS_COUNT):
        class_positions = torch.nonzero(train_labels == digit, as_tuple=False).flatten().to(torch.long)
        _require(class_positions.numel() > SAMPLER_VALIDATION_PER_CLASS + S1_NEIGHBOR_COUNT, f"class {digit} is too small")
        shuffled = class_positions[torch.randperm(class_positions.numel(), generator=generator)]
        validation = shuffled[:SAMPLER_VALIDATION_PER_CLASS].contiguous()
        fit = shuffled[SAMPLER_VALIDATION_PER_CLASS:].contiguous()
        fit_positions_by_class.append(fit)
        validation_positions_by_class.append(validation)
        va_positions_by_class.append(validation[:VALIDATION_HALF_PER_CLASS].contiguous())
        vb_positions_by_class.append(validation[VALIDATION_HALF_PER_CLASS:].contiguous())
    fit_positions = torch.cat(fit_positions_by_class)
    validation_positions = torch.cat(validation_positions_by_class)
    va_positions = torch.cat(va_positions_by_class)
    vb_positions = torch.cat(vb_positions_by_class)
    _require(torch.unique(fit_positions).numel() == fit_positions.numel(), "SAMPLER_FIT has duplicate positions")
    _require(torch.unique(torch.cat((fit_positions, validation_positions))).numel() == TRAIN_COUNT, "SAMPLER_FIT and validation overlap")
    return {
        "seed": SPLIT_SEED,
        "fit_positions": fit_positions,
        "validation_positions": validation_positions,
        "validation_a_positions": va_positions,
        "validation_b_positions": vb_positions,
        "fit_positions_by_class": fit_positions_by_class,
        "validation_positions_by_class": validation_positions_by_class,
        "validation_a_positions_by_class": va_positions_by_class,
        "validation_b_positions_by_class": vb_positions_by_class,
        "fit_source_indices": train_indices.index_select(0, fit_positions),
        "validation_source_indices": train_indices.index_select(0, validation_positions),
    }


def _split_identity(split: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split_seed": SPLIT_SEED,
        "fit_count": int(split["fit_positions"].numel()),
        "validation_count": int(split["validation_positions"].numel()),
        "validation_a_count": int(split["validation_a_positions"].numel()),
        "validation_b_count": int(split["validation_b_positions"].numel()),
        "fit_positions_sha256": tensor_sha256(split["fit_positions"]),
        "validation_positions_sha256": tensor_sha256(split["validation_positions"]),
        "validation_a_positions_sha256": tensor_sha256(split["validation_a_positions"]),
        "validation_b_positions_sha256": tensor_sha256(split["validation_b_positions"]),
        "fit_source_indices_sha256": tensor_sha256(split["fit_source_indices"]),
        "validation_source_indices_sha256": tensor_sha256(split["validation_source_indices"]),
        "per_class_fit_counts": [int(item.numel()) for item in split["fit_positions_by_class"]],
        "per_class_validation_counts": [int(item.numel()) for item in split["validation_positions_by_class"]],
        "role_contract": "SAMPLER_FIT fits candidates; SAMPLER_VALIDATION defines local-support q95 and no DEV statistic selects a candidate",
    }


def _s3_representation(z: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = statistical_variable(z, scale).to(torch.float64).cpu()
    _finite(u, "S3 statistical variable")
    _require(bool((u >= 0.0).all().item()), "S3 statistical variable left nonnegative support")
    gauged, _, anchors = max_q_gauge(z, scale)
    phase = wrap_phase(torch.angle(gauged).to(torch.float64)).cpu()
    w = torch.cat((torch.log(u + EPS_S), phase.cos(), phase.sin()), dim=1).contiguous()
    _finite(w, "S3 joint amplitude-phase representation")
    return w, u, anchors.to(torch.long).cpu()


def _fit_joint_model(x: torch.Tensor, labels: torch.Tensor, *, representation: str) -> dict[str, Any]:
    x = x.to(torch.float64).cpu().contiguous()
    labels = labels.to(torch.long).cpu().contiguous()
    dimension = int(x.shape[1])
    per_class: list[dict[str, Any]] = []
    for digit in range(CLASS_COUNT):
        rows = x[labels == digit]
        _require(rows.shape[0] > S1_NEIGHBOR_COUNT, f"class {digit} has insufficient model rows")
        mean = rows.mean(dim=0)
        centered = rows - mean
        covariance = centered.T @ centered / float(rows.shape[0])
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.flip(0).clamp_min(0.0).contiguous()
        eigenvectors = eigenvectors.flip(1).contiguous()
        total_variance = eigenvalues.sum()
        if float(total_variance) <= 0.0:
            rank = 0
        else:
            cumulative = torch.cumsum(eigenvalues, dim=0) / total_variance
            rank = int(torch.nonzero(cumulative >= PCA_VARIANCE_FRACTION, as_tuple=False)[0].item() + 1)
            rank = min(rank, PCA_RANK_CAP)
        explained = float(eigenvalues[:rank].sum() / total_variance) if float(total_variance) > 0.0 else 1.0
        remaining = max(0.0, float(total_variance) - float(eigenvalues[:rank].sum()))
        denominator = max(dimension - rank, 1)
        residual_variance = max(remaining / float(denominator), RESIDUAL_VARIANCE_FLOOR)
        per_class.append(
            {
                "mean": mean.contiguous(),
                "basis": eigenvectors[:, :rank].contiguous(),
                "eigenvalues": eigenvalues[:rank].contiguous(),
                "residual_variance": residual_variance,
                "total_variance": float(total_variance),
                "rank": rank,
                "explained_fraction": explained,
                "cap_hit": bool(rank == PCA_RANK_CAP and explained < PCA_VARIANCE_FRACTION),
                "fit_count": int(rows.shape[0]),
            }
        )
    return {
        "schema": "semantic_conditioned_joint_low_rank_model.v1",
        "representation": representation,
        "dimension": dimension,
        "variance_fraction_rule": PCA_VARIANCE_FRACTION,
        "rank_cap": PCA_RANK_CAP,
        "residual_rule": "isotropic residual variance=max(remaining within-class variance/(dimension-rank), 1e-8)",
        "per_class": per_class,
    }


def _model_summary(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "representation": model["representation"],
        "dimension": int(model["dimension"]),
        "variance_fraction_rule": float(model["variance_fraction_rule"]),
        "rank_cap": int(model["rank_cap"]),
        "ranks": [int(item["rank"]) for item in model["per_class"]],
        "explained_fraction_by_class": [float(item["explained_fraction"]) for item in model["per_class"]],
        "residual_variance_by_class": [float(item["residual_variance"]) for item in model["per_class"]],
        "cap_hit_by_class": [bool(item["cap_hit"]) for item in model["per_class"]],
    }


def _sample_joint_model(model: Mapping[str, Any], labels: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for digit, seed in zip(labels.tolist(), seeds.tolist(), strict=True):
        item = model["per_class"][int(digit)]
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        rank = int(item["rank"])
        value = item["mean"].clone()
        if rank > 0:
            coefficient = torch.randn(rank, generator=generator, dtype=torch.float64)
            value = value + item["basis"] @ (item["eigenvalues"].clamp_min(0.0).sqrt() * coefficient)
        residual = torch.randn(int(model["dimension"]), generator=generator, dtype=torch.float64)
        value = value + math.sqrt(float(item["residual_variance"])) * residual
        values.append(value)
    result = torch.stack(values).contiguous()
    _finite(result, "joint model samples")
    return result


def _make_requested_bank() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    requested = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(BANK_PER_CLASS)
    sample_indices = torch.arange(BANK_PER_CLASS, dtype=torch.long).repeat(CLASS_COUNT)
    seeds = torch.tensor(
        [BANK_SEED + int(digit) * 100_000 + int(index) for digit, index in zip(requested.tolist(), sample_indices.tolist(), strict=True)],
        dtype=torch.int64,
    )
    _require(torch.bincount(requested, minlength=CLASS_COUNT).tolist() == [BANK_PER_CLASS] * CLASS_COUNT, "requested bank is not balanced")
    return requested, sample_indices, seeds


def _s1_samples(
    fit_positions_by_class: list[torch.Tensor],
    fit_x_by_class: list[torch.Tensor],
    requested: torch.Tensor,
    sample_indices: torch.Tensor,
    seeds: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    anchors_local: list[int] = []
    lambdas: list[float] = []
    for digit, seed in zip(requested.tolist(), seeds.tolist(), strict=True):
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        count = int(fit_x_by_class[int(digit)].shape[0])
        anchors_local.append(int(torch.randint(count, (1,), generator=generator)))
        lambdas.append(float(torch.rand((), generator=generator, dtype=torch.float64)))
    neighbors_local: list[int] = [0] * BANK_COUNT
    for digit in range(CLASS_COUNT):
        query_indices = [row for row, value in enumerate(requested.tolist()) if value == digit]
        unique_anchors = sorted(set(anchors_local[row] for row in query_indices))
        query = fit_x_by_class[digit].index_select(0, torch.tensor(unique_anchors, dtype=torch.long))
        distances = torch.cdist(query, fit_x_by_class[digit], p=2)
        for row, anchor in enumerate(unique_anchors):
            distances[row, anchor] = float("inf")
        nearest = distances.topk(S1_NEIGHBOR_COUNT, largest=False, dim=1).indices
        lookup = {anchor: nearest[row] for row, anchor in enumerate(unique_anchors)}
        for row in query_indices:
            rank = (digit * BANK_PER_CLASS + int(sample_indices[row])) % S1_NEIGHBOR_COUNT
            neighbors_local[row] = int(lookup[anchors_local[row]][rank])
    samples: list[torch.Tensor] = []
    anchor_fit_indices: list[int] = []
    neighbor_fit_indices: list[int] = []
    for row, digit in enumerate(requested.tolist()):
        anchor = anchors_local[row]
        neighbor = neighbors_local[row]
        lam = lambdas[row]
        x = (1.0 - lam) * fit_x_by_class[digit][anchor] + lam * fit_x_by_class[digit][neighbor]
        samples.append(x)
        anchor_fit_indices.append(int(fit_positions_by_class[digit][anchor]))
        neighbor_fit_indices.append(int(fit_positions_by_class[digit][neighbor]))
    result = torch.stack(samples).contiguous()
    _finite(result, "S1 samples")
    return result, {
        "family": "S1",
        "anchor_rule": "deterministic per-row CPU seed selects one SAMPLER_FIT_y anchor",
        "neighborhood_rule": "exact same-class Euclidean 16-nearest-neighbor set in global TRAIN-FIT standardized Cartesian OLS",
        "neighbor_rule": "rank=(class*200+sample_index) mod 16",
        "interpolation_rule": "x=(1-lambda)*anchor+lambda*neighbor with lambda~Uniform(0,1) from the same per-row seed",
        "anchor_train_positions": anchor_fit_indices,
        "neighbor_train_positions": neighbor_fit_indices,
        "lambda": lambdas,
    }


def _reconstruct_s3(value: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = value[:, :OLS_PORT_COUNT]
    u = torch.exp(h) - EPS_S
    _finite(u, "S3 reconstructed amplitude variable")
    _require(bool((u >= 0.0).all().item()), "S3 reconstruction produced negative amplitude support")
    cosine = value[:, OLS_PORT_COUNT : 2 * OLS_PORT_COUNT]
    sine = value[:, 2 * OLS_PORT_COUNT :]
    norm = torch.sqrt(cosine.square() + sine.square())
    _finite(norm, "S3 phase-pair norm")
    _require(bool((norm > 0.0).all().item()), "S3 reconstruction has a zero circular phase pair")
    phase = wrap_phase(torch.atan2(sine / norm, cosine / norm))
    amplitude = ((scale + EPS_S) * u).sqrt()
    z_before_gauge = torch.complex(amplitude * phase.cos(), amplitude * phase.sin()).to(torch.complex64)
    gauged, _, _ = max_q_gauge(z_before_gauge, scale)
    reconstructed_u = statistical_variable(gauged, scale).to(torch.float64).cpu()
    _require(torch.allclose(reconstructed_u, u, rtol=2.0e-5, atol=2.0e-6), "S3 gauge reconstruction changed amplitude semantics")
    _finite(gauged.real, "S3 reconstructed OLS real part")
    _finite(gauged.imag, "S3 reconstructed OLS imaginary part")
    return gauged.cpu().contiguous(), reconstructed_u


def _bootstrap_rates(correct: torch.Tensor, requested: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    correct = correct.to(torch.float64).reshape(-1, BANK_COUNT)
    requested = requested.reshape(-1)
    _require(correct.shape[1] == BANK_COUNT, "bootstrap bank shape changed")
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    draws = torch.randint(BANK_PER_CLASS, (BOOTSTRAP_COUNT, CLASS_COUNT, BANK_PER_CLASS), generator=generator)
    per_class = correct.reshape(correct.shape[0], CLASS_COUNT, BANK_PER_CLASS)
    expanded = per_class[:, None, :, :].expand(correct.shape[0], BOOTSTRAP_COUNT, CLASS_COUNT, BANK_PER_CLASS)
    gather_index = draws[None, :, :, :].expand(correct.shape[0], BOOTSTRAP_COUNT, CLASS_COUNT, BANK_PER_CLASS)
    sampled = expanded.gather(3, gather_index).mean(dim=3).mean(dim=2)
    observed = correct.reshape(correct.shape[0], CLASS_COUNT, BANK_PER_CLASS).mean(dim=2).mean(dim=1)
    del requested
    return observed, sampled


def _ci(values: torch.Tensor) -> dict[str, Any]:
    values = values.to(torch.float64).cpu()
    quantiles = torch.quantile(values, torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64))
    return {"lower": float(quantiles[0]), "median": float(quantiles[1]), "upper": float(quantiles[2]), "level": 0.95}


def _support_descriptive(
    candidate_u: torch.Tensor,
    train_u: torch.Tensor,
    real_a_u: torch.Tensor,
    real_b_u: torch.Tensor,
) -> dict[str, Any]:
    log_train = torch.log(train_u.to(torch.float64) + EPS_S)
    h_mean = log_train.mean(dim=0)
    h_std = log_train.std(dim=0, unbiased=False).clamp_min(RESIDUAL_VARIANCE_FLOOR**0.5)
    sample_h = (torch.log(candidate_u.to(torch.float64) + EPS_S) - h_mean) / h_std
    real_a_h = (torch.log(real_a_u.to(torch.float64) + EPS_S) - h_mean) / h_std
    real_b_h = (torch.log(real_b_u.to(torch.float64) + EPS_S) - h_mean) / h_std
    cross = canary._energy_distance(sample_h, real_a_h)
    reference = canary._energy_distance(real_a_h, real_b_h)
    ranges = ((candidate_u >= train_u.amin(dim=0)) & (candidate_u <= train_u.amax(dim=0))).to(torch.float64).mean()
    return {
        "R_u": cross / max(reference, 1.0e-12),
        "energy_distance_candidate_vs_validation_A": cross,
        "energy_distance_validation_A_vs_validation_B": reference,
        "fraction_inside_frozen_real_train_port_ranges": float(ranges),
        "log_amplitude_standardization": "h=log(u+1e-12), mean/std fit on all real TRAIN OLS rows",
        "log_amplitude_candidate_summary": _summary(sample_h),
        "log_amplitude_validation_A_summary": _summary(real_a_h),
        "log_amplitude_covariance": canary._covariance_comparison(sample_h, real_a_h),
    }


def _phase_descriptive(candidate_z: torch.Tensor, real_a_z: torch.Tensor, real_b_z: torch.Tensor, scale: torch.Tensor) -> dict[str, Any]:
    candidate_gauged, _, candidate_anchor = max_q_gauge(candidate_z, scale)
    real_a_gauged, _, real_a_anchor = max_q_gauge(real_a_z, scale)
    real_b_gauged, _, real_b_anchor = max_q_gauge(real_b_z, scale)
    return canary._phase_distribution_comparison(
        wrap_phase(torch.angle(candidate_gauged).to(torch.float64)),
        wrap_phase(torch.angle(real_a_gauged).to(torch.float64)),
        wrap_phase(torch.angle(real_b_gauged).to(torch.float64)),
        candidate_anchor,
        real_a_anchor,
        real_b_anchor,
    )


def _raw_state() -> dict[str, Any]:
    return {"count": 0, "finite": True, "nonconstant": [], "zero": [], "minimum": [], "maximum": [], "total": [], "mean": [], "std": []}


def _update_raw_state(state: dict[str, Any], raw: torch.Tensor) -> None:
    _require(raw.ndim == 4 and tuple(raw.shape[1:]) == (1, 200, 200), "raw CMOS geometry changed")
    _finite(raw, "raw CMOS")
    flat = raw.detach().cpu().to(torch.float64).flatten(1)
    per_std = flat.std(dim=1, unbiased=False)
    state["count"] += int(flat.shape[0])
    state["finite"] = bool(state["finite"] and torch.isfinite(raw).all().item())
    state["nonconstant"].extend((per_std > 0.0).tolist())
    state["zero"].extend((flat.abs().sum(dim=1) == 0.0).tolist())
    state["minimum"].extend(flat.min(dim=1).values.tolist())
    state["maximum"].extend(flat.max(dim=1).values.tolist())
    state["total"].extend(flat.sum(dim=1).tolist())
    state["mean"].extend(flat.mean(dim=1).tolist())
    state["std"].extend(per_std.tolist())


def _finish_raw_state(state: Mapping[str, Any]) -> dict[str, Any]:
    count = int(state["count"])
    per_std = torch.tensor(state["std"], dtype=torch.float64)
    per_min = torch.tensor(state["minimum"], dtype=torch.float64)
    per_max = torch.tensor(state["maximum"], dtype=torch.float64)
    total = torch.tensor(state["total"], dtype=torch.float64)
    mean = torch.tensor(state["mean"], dtype=torch.float64)
    _require(count > 0, "raw CMOS stream is empty")
    return {
        "shape": [count, 1, 200, 200],
        "dtype": "torch.float32",
        "finite": bool(state["finite"]),
        "nonconstant_fraction": float(torch.tensor(state["nonconstant"], dtype=torch.float64).mean()),
        "nonconstant_sample_count": int(sum(bool(item) for item in state["nonconstant"])),
        "zero_image_fraction": float(torch.tensor(state["zero"], dtype=torch.float64).mean()),
        "zero_image_count": int(sum(bool(item) for item in state["zero"])),
        "intensity_mean_per_sample": _summary(mean),
        "per_sample_std": _summary(per_std),
        "per_sample_min": _summary(per_min),
        "per_sample_max": _summary(per_max),
        "total_intensity": _summary(total),
    }


def _panel_indices() -> torch.Tensor:
    return torch.cat([torch.arange(digit * BANK_PER_CLASS, digit * BANK_PER_CLASS + METRIC_PER_CLASS) for digit in range(CLASS_COUNT)]).to(torch.long)


def _run_canary_bank(
    consumer: Any,
    models: Mapping[str, torch.nn.Module],
    z: torch.Tensor,
    device: torch.device,
    *,
    panel_indices: torch.Tensor,
) -> dict[str, Any]:
    raw_state = _raw_state()
    panel_set = set(int(item) for item in panel_indices.tolist())
    raw_panel: dict[int, torch.Tensor] = {}
    embeddings: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    logits: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    for start in range(0, z.shape[0], CANARY_BATCH_SIZE):
        stop = min(start + CANARY_BATCH_SIZE, z.shape[0])
        with torch.inference_mode():
            result = consumer.generated_forward(z[start:stop].to(device))
            raw = result.raw_intensity.detach().cpu()
            view = F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0).to(device), (28, 28))
            evaluator_outputs = {name: model(view) for name, model in models.items()}
        _update_raw_state(raw_state, raw)
        for name, (features, output_logits) in evaluator_outputs.items():
            _finite(features, f"{name} features")
            _finite(output_logits, f"{name} logits")
            embeddings[name].append(features.detach().cpu())
            logits[name].append(output_logits.detach().cpu())
        for local, global_index in enumerate(range(start, stop)):
            if global_index in panel_set:
                raw_panel[global_index] = raw[local].clone()
        if stop == z.shape[0] or stop % 200 == 0:
            print(json.dumps({"phase": "GENERATOR_CANARY", "completed": stop, "total": int(z.shape[0])}, ensure_ascii=False), flush=True)
    ordered_panel = torch.cat([raw_panel[int(index)] for index in panel_indices.tolist()], dim=0)
    return {
        "raw_stats": _finish_raw_state(raw_state),
        "raw_panel": ordered_panel,
        "embeddings": {name: torch.cat(values, dim=0) for name, values in embeddings.items()},
        "logits": {name: torch.cat(values, dim=0) for name, values in logits.items()},
    }


def _canary_semantic_report(output: Mapping[str, Any], requested: torch.Tensor) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, logits in output["logits"].items():
        classification, predictions, _ = _classification_report(logits, requested)
        correct = predictions == requested
        _, bootstrap = _bootstrap_rates(correct.to(torch.float64), requested)
        report[name] = {
            **classification,
            "macro_recall_bootstrap_ci": _ci(bootstrap[0]),
            "requested_vs_output_predictions": {
                "macro_recall": classification["macro_recall"],
                "predicted_class_histogram": classification["predicted_class_histogram"],
            },
        }
    return report


def _canary_comparison(output: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    generated_raw = output["raw_panel"].flatten(1).to(torch.float64)
    control_raw = control["raw_panel"].flatten(1).to(torch.float64)
    raw_mean = control_raw.mean(dim=0)
    raw_std = control_raw.std(dim=0, unbiased=False).clamp_min(1.0e-8)
    generated_raw_standardized = (generated_raw - raw_mean) / raw_std
    control_raw_standardized = (control_raw - raw_mean) / raw_std
    raw_cross = canary._energy_distance(generated_raw_standardized, control_raw_standardized)
    raw_self = canary._energy_distance(control_raw_standardized[:100], control_raw_standardized[100:])
    generated_feature = output["embeddings"]["E_C"].to(torch.float64)
    control_feature = control["embeddings"]["E_C"].to(torch.float64)
    feature_cross = canary._energy_distance(generated_feature, control_feature)
    feature_self = canary._energy_distance(control_feature[:100], control_feature[100:])
    return {
        "R_raw": raw_cross / max(raw_self, 1.0e-12),
        "raw_energy_distance_generated_vs_control": raw_cross,
        "raw_energy_distance_control_first100_vs_second100": raw_self,
        "R_feature": feature_cross / max(feature_self, 1.0e-12),
        "feature_energy_distance_generated_vs_control": feature_cross,
        "feature_energy_distance_control_first100_vs_second100": feature_self,
        "raw_standardization": "per-pixel mean/std fit on the fixed 200-row real validation control panel",
        "feature_space": "frozen E_C embedding",
    }


def _canary_diversity(z: torch.Tensor, output: Mapping[str, Any]) -> dict[str, Any]:
    raw_view = F.adaptive_avg_pool2d(output["raw_panel"].clamp(0.0, 1.0), (28, 28))
    return {
        "latent_z": canary._diversity_view(z, name="canary_latent_z"),
        "raw_cmos_metric_panel_28x28": canary._diversity_view(raw_view, name="canary_raw_cmos_metric_panel_28x28"),
        "E_C_feature": canary._diversity_view(output["embeddings"]["E_C"], name="canary_E_C_feature"),
    }


def _load_authorities(root: Path) -> dict[str, Any]:
    git = robust.git_identity(root)
    _require(git["branch"] == EXPECTED_BRANCH and git["head"] == EXPECTED_HEAD, "branch/HEAD conflicts with accepted authorities")
    robust_authorities, run_inputs, preflight = readiness._load_authorities_and_preflight(root)
    del run_inputs

    train_path = root / TRAIN_OLS
    dev_path = root / DEV_OLS
    _require(file_sha256(train_path) == TRAIN_OLS_SHA256, "accepted TRAIN OLS identity changed")
    _require(file_sha256(dev_path) == DEV_OLS_SHA256, "accepted DEV OLS identity changed")
    train_payload = _mapping(safe_torch_load(train_path, expected_sha256=TRAIN_OLS_SHA256, map_location="cpu"), "accepted TRAIN OLS")
    dev_payload = _mapping(safe_torch_load(dev_path, expected_sha256=DEV_OLS_SHA256, map_location="cpu"), "accepted DEV OLS")
    train_z = _tensor(train_payload.get("z_final"), "accepted TRAIN z")
    dev_z = _tensor(dev_payload.get("z_dev"), "accepted DEV z")
    train_indices = _tensor(train_payload.get("source_indices"), "accepted TRAIN source indices").to(torch.long)
    dev_indices = _tensor(dev_payload.get("source_indices"), "accepted DEV source indices").to(torch.long)
    labels = _tensor(robust_authorities["labels"], "canonical labels").to(torch.long)
    canonical_train_indices = _tensor(robust_authorities["train_indices"], "canonical TRAIN source indices").to(torch.long)
    canonical_dev_indices = _tensor(robust_authorities["development_indices"], "canonical DEV source indices").to(torch.long)
    _require(torch.equal(train_indices, canonical_train_indices), "TRAIN OLS source identity differs from accepted split")
    _require(torch.equal(dev_indices, canonical_dev_indices), "DEV OLS source identity differs from accepted split")
    _require(train_z.shape == (TRAIN_COUNT, OLS_PORT_COUNT) and train_z.dtype == torch.complex64, "TRAIN OLS shape/dtype changed")
    _require(dev_z.shape == (DEV_COUNT, OLS_PORT_COUNT) and dev_z.dtype == torch.complex64, "DEV OLS shape/dtype changed")
    _finite(train_z.real, "TRAIN OLS real part")
    _finite(train_z.imag, "TRAIN OLS imaginary part")
    _finite(dev_z.real, "DEV OLS real part")
    _finite(dev_z.imag, "DEV OLS imaginary part")
    train_labels = labels.index_select(0, train_indices)
    dev_labels = labels.index_select(0, dev_indices)
    _require(torch.bincount(train_labels, minlength=CLASS_COUNT).tolist() == robust_authorities["data_identity"]["train_counts_by_class"], "TRAIN labels changed")
    _require(torch.bincount(dev_labels, minlength=CLASS_COUNT).tolist() == [500] * CLASS_COUNT, "DEV labels changed")

    probe = gap._load_probe(root)
    _require(file_sha256(root / PROBE_WEIGHTS) == PROBE_WEIGHTS_SHA256, "accepted probe identity changed")
    scale = _tensor(robust_authorities["cpu_context"]["scale"], "accepted OLS scale").to(torch.float64).cpu()
    _require(scale.shape == (OLS_PORT_COUNT,), "accepted OLS scale shape changed")
    _finite(scale, "accepted OLS scale")
    _require(bool((scale > EPS_S).all().item()), "accepted OLS scale is not strictly positive")

    p2_path = root / P2Q2_BANK
    _require(file_sha256(p2_path) == P2Q2_BANK_SHA256, "fixed P2/Q2 bank identity changed")
    p2_payload = _mapping(safe_torch_load(p2_path, expected_sha256=P2Q2_BANK_SHA256, map_location="cpu"), "fixed P2/Q2 bank")
    p2_z = _tensor(p2_payload.get("z_sample"), "fixed P2/Q2 z")
    _require(p2_payload.get("candidate_id") == "Q2", "fixed P2/Q2 candidate identity changed")
    _require(p2_z.shape == (1000, OLS_PORT_COUNT) and p2_z.dtype == torch.complex64, "fixed P2/Q2 bank shape/dtype changed")
    _finite(p2_z.real, "fixed P2/Q2 real part")
    _finite(p2_z.imag, "fixed P2/Q2 imaginary part")

    gap_path = root / GAP_REPORT
    _require(file_sha256(gap_path) == GAP_REPORT_SHA256, "accepted manifold-gap report identity changed")
    gap_report = json.loads(gap_path.read_text(encoding="utf-8"))
    _require(gap_report.get("result") == "VALID_COMPLETED", "accepted manifold-gap result changed")
    _require(gap_report.get("outer_gate") == "REAL_VS_SAMPLED_OLS_SEMANTIC_MANIFOLD_GAP_LOCALIZED", "accepted manifold-gap gate changed")
    _require(gap_report.get("classification") == SUPPORTED_BOTTLENECK, "accepted manifold-gap classification changed")

    checkpoint = readiness._checkpoint_payload(root, {"step": 3000, **readiness.REAL_CHECKPOINTS[3000]})
    _require(checkpoint.get("completed_updates") == 3000, "step3000 checkpoint changed")

    # The authority loader has already verified these files; release image
    # memory before the statistical fit, because the sampler consumes OLS only.
    robust_authorities.pop("images", None)
    return {
        "git": git,
        "robust": robust_authorities,
        "preflight": preflight,
        "train_z": train_z.cpu().contiguous(),
        "dev_z": dev_z.cpu().contiguous(),
        "train_indices": train_indices.cpu().contiguous(),
        "dev_indices": dev_indices.cpu().contiguous(),
        "labels": labels.cpu().contiguous(),
        "train_labels": train_labels.cpu().contiguous(),
        "dev_labels": dev_labels.cpu().contiguous(),
        "probe": probe,
        "scale": scale,
        "p2_path": p2_path,
        "p2_z": p2_z.cpu().contiguous(),
        "gap_path": gap_path,
        "gap_report": gap_report,
        "checkpoint": checkpoint,
    }


def _latent_support_report(
    name: str,
    z: torch.Tensor,
    requested: torch.Tensor,
    sample_indices: torch.Tensor,
    fit_x: torch.Tensor,
    fit_x_by_class: list[torch.Tensor],
    validation_x_by_class: list[torch.Tensor],
    q95_by_class: torch.Tensor,
    fit_mean: torch.Tensor,
    fit_std: torch.Tensor,
    train_z: torch.Tensor,
    train_labels: torch.Tensor,
    probe: Mapping[str, Any],
    probe_mean: torch.Tensor,
    probe_std: torch.Tensor,
    train_logits: torch.Tensor,
    train_u: torch.Tensor,
    real_a_z: torch.Tensor,
    real_b_z: torch.Tensor,
    scale: torch.Tensor,
    metric_indices: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cart = _ols_features(z)
    x = _apply_standardizer(cart, fit_mean, fit_std)
    probe_logits = _probe_logits(z, probe_mean, probe_std, probe)
    probe_report, predictions, probabilities = _classification_report(probe_logits, requested)
    success = torch.zeros(BANK_COUNT, dtype=torch.bool)
    target_distances = torch.empty(BANK_COUNT, dtype=torch.float64)
    for digit in range(CLASS_COUNT):
        rows = torch.nonzero(requested == digit, as_tuple=False).flatten()
        distances, _ = _nearest_distance(x.index_select(0, rows), fit_x_by_class[digit])
        target_distances.index_copy_(0, rows, distances)
        success.index_copy_(0, rows, (predictions.index_select(0, rows) == digit) & (distances <= q95_by_class[digit]))
    metric_x = x.index_select(0, metric_indices)
    all_class_distances = torch.stack([_nearest_distance(metric_x, fit_x_by_class[digit])[0] for digit in range(CLASS_COUNT)], dim=1)
    nearest_class = all_class_distances.argmin(dim=1)
    metric_requested = requested.index_select(0, metric_indices)
    latent_train_centroids = torch.stack([train_logits[train_labels == digit].mean(dim=0) for digit in range(CLASS_COUNT)])
    metric_logits = probe_logits.index_select(0, metric_indices)
    centroid_distances = torch.cdist(metric_logits, latent_train_centroids, p=2)
    centroid_nearest = centroid_distances.argmin(dim=1)
    sample_u = statistical_variable(z, scale).to(torch.float64).cpu()
    real_a_u = statistical_variable(real_a_z, scale).to(torch.float64).cpu()
    real_b_u = statistical_variable(real_b_z, scale).to(torch.float64).cpu()
    support = _support_descriptive(sample_u.index_select(0, metric_indices), train_u, real_a_u, real_b_u)
    complex_report, _, _ = canary._complex_comparison(z.index_select(0, metric_indices), real_a_z, real_b_z, train_z)
    phase_report = _phase_descriptive(z.index_select(0, metric_indices), real_a_z, real_b_z, scale)
    probe_ordered = torch.sort(probabilities, dim=1, descending=True).values
    per_class_support = [float(success[requested == digit].to(torch.float64).mean()) for digit in range(CLASS_COUNT)]
    observed, bootstrap = _bootstrap_rates(success.to(torch.float64).unsqueeze(0), requested)
    semantic_rate = float(observed[0])
    report = {
        "family": name,
        "rows": BANK_COUNT,
        "requested_class_histogram": _int_list(torch.bincount(requested, minlength=CLASS_COUNT)),
        "semantic_support_rate": semantic_rate,
        "semantic_support_rate_bootstrap_ci": _ci(bootstrap[0]),
        "per_class_semantic_support_rate": per_class_support,
        "semantic_support_definition": "frozen OLS probe prediction equals requested y AND same-class nearest SAMPLER_FIT_y distance <= TRAIN-validation class-y q95",
        "class_q95_local_support_radius": _float_list(q95_by_class),
        "probe": {
            **probe_report,
            "requested_class_confidence": _float_list(probabilities.gather(1, requested[:, None]).squeeze(1)),
            "requested_class_top1_fraction": float((predictions == requested).to(torch.float64).mean()),
            "top1_top2_margin": _summary(probe_ordered[:, 0] - probe_ordered[:, 1]),
        },
        "local_support": {
            "same_class_nearest_distance": _summary(target_distances),
            "same_class_nearest_distance_by_requested_class": [_summary(target_distances[requested == digit]) for digit in range(CLASS_COUNT)],
            "class_support_radius_source": "SAMPLER_VALIDATION rows only; no DEV threshold used",
            "metric_panel_nearest_real_class_agreement": float((nearest_class == metric_requested).to(torch.float64).mean()),
            "metric_panel_nearest_real_distance": _summary(all_class_distances.min(dim=1).values),
        },
        "real_class_centroid_logit_geometry": {
            "metric_panel_nearest_centroid_requested_class_agreement": float((centroid_nearest == metric_requested).to(torch.float64).mean()),
            "distance_to_requested_centroid": _summary(centroid_distances.gather(1, metric_requested[:, None]).squeeze(1)),
            "distance_to_nearest_centroid": _summary(centroid_distances.min(dim=1).values),
        },
        "R_u_and_amplitude": support,
        "R_z_and_joint_complex": complex_report,
        "phase_compatibility": phase_report,
        "finite": bool(torch.isfinite(z.real).all().item() and torch.isfinite(z.imag).all().item()),
        "nonconstant_fraction": float((_ols_features(z).std(dim=1, unbiased=False) > 0.0).to(torch.float64).mean()),
        "generator_not_used_for_selection": True,
    }
    internal = {
        "z": z,
        "u": sample_u,
        "success": success,
        "probe_predictions": predictions,
        "report": report,
    }
    return report, internal


def _select_winner(candidate_reports: Mapping[str, Mapping[str, Any]], successes: Mapping[str, torch.Tensor], requested: torch.Tensor) -> tuple[str, dict[str, Any]]:
    names = ["S1", "S2", "S3"]
    correctness = torch.stack([successes[name].to(torch.float64) for name in names])
    observed, bootstrap = _bootstrap_rates(correctness, requested)
    rates = {name: float(observed[index]) for index, name in enumerate(names)}
    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(names):
        for right_index in range(left_index + 1, len(names)):
            right = names[right_index]
            difference = bootstrap[left_index] - bootstrap[right_index]
            pairwise[f"{left}_vs_{right}"] = {
                "observed_difference": rates[left] - rates[right],
                "bootstrap_ci_difference_left_minus_right": _ci(difference),
                "bootstrap_count": BOOTSTRAP_COUNT,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "paired_requested_class_and_sample_index_resampling": True,
            }
    top = sorted(names, key=lambda name: (-rates[name], SIMPLICITY_ORDER[name]))[0]
    tied_with_top = [top]
    for name in names:
        if name == top:
            continue
        key = f"{top}_vs_{name}" if SIMPLICITY_ORDER[top] < SIMPLICITY_ORDER[name] else f"{name}_vs_{top}"
        ci = pairwise[key]["bootstrap_ci_difference_left_minus_right"]
        if (ci["lower"] <= 0.0 <= ci["upper"]):
            tied_with_top.append(name)
    winner = sorted(tied_with_top, key=lambda name: SIMPLICITY_ORDER[name])[0]
    return winner, {
        "eligible_families": names,
        "observed_semantic_support_rate": rates,
        "pairwise_bootstrap_comparisons": pairwise,
        "top_observed_family": top,
        "effectively_tied_with_top": sorted(tied_with_top, key=lambda name: SIMPLICITY_ORDER[name]),
        "winner": winner,
        "tie_rule": "highest macro semantic_support_rate; if the top-vs-candidate paired 95% CI contains zero, treat that pair as tied and select the simplest tied family S1>S2>S3",
        "no_R_z_or_Generator_selection": True,
    }


def _classification(
    winner: str,
    winner_report: Mapping[str, Any],
    s0_report: Mapping[str, Any],
    candidate_reports: Mapping[str, Mapping[str, Any]],
    winner_canary: Mapping[str, Any],
    s0_canary: Mapping[str, Any],
) -> tuple[str, str, str]:
    winner_latent = float(winner_report["semantic_support_rate"])
    winner_latent_ci = float(winner_report["semantic_support_rate_bootstrap_ci"]["lower"])
    winner_ec = float(winner_canary["semantic"]["E_C"]["macro_recall"])
    winner_ec_ci = float(winner_canary["semantic"]["E_C"]["macro_recall_bootstrap_ci"]["lower"])
    s0_latent = float(s0_report["semantic_support_rate"])
    s0_ec = float(s0_canary["semantic"]["E_C"]["macro_recall"])
    winner_optical = bool(winner_canary["optical_validity"]["passes"])
    winner_latent_pass = winner_latent >= LATENT_TARGET and winner_latent_ci >= LATENT_TARGET
    winner_ec_pass = winner_ec >= GENERATOR_TARGET and winner_ec_ci >= GENERATOR_TARGET
    s0_ec_pass = s0_ec >= GENERATOR_TARGET and float(s0_canary["semantic"]["E_C"]["macro_recall_bootstrap_ci"]["lower"]) >= GENERATOR_TARGET
    generative_max = max(float(candidate_reports[name]["semantic_support_rate"]) for name in ("S1", "S2", "S3"))
    if winner_latent_pass and winner_ec_pass and winner_optical:
        return (
            "SEMANTICALLY_CONDITIONED_OLS_SAMPLER_SUPPORTED",
            "The frozen generative winner reaches the preregistered local semantic-support target and preserves requested class semantics through E_C with valid nonconstant CMOS output.",
            "ASSESS_SEMANTICALLY_CONDITIONED_GENERATION_QUALITY_V1",
        )
    if winner_latent_pass and s0_ec_pass and (not winner_ec_pass or winner_ec < s0_ec):
        return (
            "SEMANTIC_OLS_SUPPORT_REPAIRED_BUT_GENERATOR_TRANSFER_MIXED",
            "Latent semantic support is repaired and the real-support S0 control transfers through E_C, but the selected generative sampler does not match that downstream transfer.",
            "ATTRIBUTE_GENERATIVE_SAMPLE_VS_REAL_OLS_LOCAL_JACOBIAN_GAP_V1",
        )
    if s0_latent >= LATENT_TARGET and s0_ec_pass and generative_max < LATENT_TARGET:
        return (
            "EMPIRICAL_REAL_OLS_SUPPORT_WORKS_BUT_GENERATIVE_MODELING_GAP_REMAINS",
            "Exact class-conditioned real OLS replay reaches the latent and downstream controls, while all tested generative families remain below the latent target.",
            "DEFINE_NEXT_GENERATION_OLS_MANIFOLD_MODEL_V1",
        )
    if s0_latent >= LATENT_TARGET and not s0_ec_pass:
        return (
            "REAL_OLS_REPLAY_DOES_NOT_RECOVER_EXPECTED_GENERATOR_SEMANTICS",
            "The class-conditioned real OLS replay is locally supported but the frozen step3000 Generator does not preserve requested semantics under the E_C canary.",
            "ATTRIBUTE_REAL_OLS_REPLAY_CONDITIONING_IDENTITY_GAP_V1",
        )
    if generative_max <= OLD_Q2_PREDCLASS_SUPPORT:
        return (
            "SEMANTICALLY_CONDITIONED_OLS_SAMPLER_NOT_SUPPORTED",
            "None of S1-S3 materially improves latent semantic support over the accepted P2/Q2 predicted-class support control.",
            "NONE",
        )
    return (
        "SEMANTICALLY_CONDITIONED_OLS_SAMPLER_MIXED",
        "The bounded sampler comparison produces partial latent support improvement without a clean preregistered A-D interpretation.",
        "NONE",
    )


def _run_experiment(root: Path, output_dir: Path) -> dict[str, Any]:
    outputs_root = (root / "outputs").resolve()
    target = (root / output_dir if not output_dir.is_absolute() else output_dir).resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs/")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "protected output namespace is forbidden")
    _require(not target.exists(), f"write-once output already exists: {target}")
    print(json.dumps({"phase": "AUTHORITY_PREFLIGHT", "status": "START"}, ensure_ascii=False), flush=True)
    inputs = _load_authorities(root)
    train_z = inputs["train_z"]
    dev_z = inputs["dev_z"]
    train_labels = inputs["train_labels"]
    dev_labels = inputs["dev_labels"]
    scale = inputs["scale"]
    split = _make_sampler_split(train_labels, inputs["train_indices"])
    split_identity = _split_identity(split)
    requested, sample_indices, seeds = _make_requested_bank()
    metric_indices = _panel_indices()

    train_cart = _ols_features(train_z)
    fit_cart = train_cart.index_select(0, split["fit_positions"])
    fit_x, fit_mean, fit_std = _fit_standardizer(fit_cart)
    validation_cart = train_cart.index_select(0, split["validation_positions"])
    validation_x = _apply_standardizer(validation_cart, fit_mean, fit_std)
    fit_labels = train_labels.index_select(0, split["fit_positions"])
    validation_labels = train_labels.index_select(0, split["validation_positions"])
    fit_x_by_class = [fit_x[fit_labels == digit].contiguous() for digit in range(CLASS_COUNT)]
    validation_x_by_class = [validation_x[validation_labels == digit].contiguous() for digit in range(CLASS_COUNT)]
    q95_by_class = torch.empty(CLASS_COUNT, dtype=torch.float64)
    for digit in range(CLASS_COUNT):
        distances, _ = _nearest_distance(validation_x_by_class[digit], fit_x_by_class[digit])
        q95_by_class[digit] = torch.quantile(distances, torch.tensor(0.95, dtype=torch.float64))
    _finite(q95_by_class, "class local-support q95")

    _, probe_mean_fit, probe_std_fit = _fit_standardizer(train_cart)
    _require(tensor_sha256(probe_mean_fit) == PROBE_MEAN_SHA256, "accepted OLS probe mean reconstruction changed")
    _require(tensor_sha256(probe_std_fit) == PROBE_STD_SHA256, "accepted OLS probe std reconstruction changed")
    probe_train_logits = _probe_logits(train_z, probe_mean_fit, probe_std_fit, inputs["probe"])
    probe_dev_logits = _probe_logits(dev_z, probe_mean_fit, probe_std_fit, inputs["probe"])
    dev_probe_report, _, _ = _classification_report(probe_dev_logits, dev_labels)
    _require(abs(float(dev_probe_report["macro_recall"]) - 0.9226) < 1.0e-12, "accepted DEV probe qualification did not reproduce")

    train_u = statistical_variable(train_z, scale).to(torch.float64).cpu()
    s3_train_w, _, _ = _s3_representation(train_z, scale)
    fit_s3_w = s3_train_w.index_select(0, split["fit_positions"])
    s3_fit_x, s3_mean, s3_std = _fit_standardizer(fit_s3_w)
    s3_fit_labels = fit_labels.clone()
    s2_model = _fit_joint_model(fit_x, fit_labels, representation="[Re(z),Im(z)] in R^200")
    s3_model = _fit_joint_model(s3_fit_x, s3_fit_labels, representation="[log(u),cos(phi_gauge),sin(phi_gauge)] in R^300")

    s0_z = torch.cat([train_z.index_select(0, split["fit_positions_by_class"][digit][:BANK_PER_CLASS]) for digit in range(CLASS_COUNT)]).contiguous()
    s1_x, s1_sampling = _s1_samples(
        split["fit_positions_by_class"],
        fit_x_by_class,
        requested,
        sample_indices,
        seeds,
    )
    s1_z = _complex_from_cartesian(s1_x * fit_std + fit_mean)
    s2_x = _sample_joint_model(s2_model, requested, seeds)
    s2_z = _complex_from_cartesian(s2_x * fit_std + fit_mean)
    s3_sample_w = _sample_joint_model(s3_model, requested, seeds)
    s3_z, s3_u = _reconstruct_s3(s3_sample_w * s3_std + s3_mean, scale)
    candidate_z = {"S0": s0_z, "S1": s1_z, "S2": s2_z, "S3": s3_z}
    for name, value in candidate_z.items():
        _require(value.shape == (BANK_COUNT, OLS_PORT_COUNT) and value.dtype == torch.complex64, f"{name} bank shape/dtype changed")
        _finite(value.real, f"{name} real part")
        _finite(value.imag, f"{name} imaginary part")
        _require(bool((_ols_features(value).std(dim=1, unbiased=False) > 0.0).all().item()), f"{name} bank contains constant OLS rows")

    real_a_z = train_z.index_select(0, split["validation_a_positions"])
    real_b_z = train_z.index_select(0, split["validation_b_positions"])
    candidate_reports: dict[str, dict[str, Any]] = {}
    candidate_internal: dict[str, dict[str, Any]] = {}
    for name in ("S0", "S1", "S2", "S3"):
        report, internal = _latent_support_report(
            name,
            candidate_z[name],
            requested,
            sample_indices,
            fit_x,
            fit_x_by_class,
            validation_x_by_class,
            q95_by_class,
            fit_mean,
            fit_std,
            train_z,
            train_labels,
            inputs["probe"],
            probe_mean_fit,
            probe_std_fit,
            probe_train_logits,
            train_u,
            real_a_z,
            real_b_z,
            scale,
            metric_indices,
        )
        candidate_reports[name] = report
        candidate_internal[name] = internal
        print(json.dumps({"phase": "LATENT_CANDIDATE", "family": name, "semantic_support_rate": report["semantic_support_rate"]}, ensure_ascii=False), flush=True)

    winner, selection = _select_winner(
        {name: candidate_reports[name] for name in ("S1", "S2", "S3")},
        {name: candidate_internal[name]["success"] for name in ("S1", "S2", "S3")},
        requested,
    )
    winner_z = candidate_z[winner]
    winner_report = candidate_reports[winner]
    s0_report = candidate_reports["S0"]
    print(json.dumps({"phase": "WINNER_FROZEN", "winner": winner, "semantic_support_rate": winner_report["semantic_support_rate"]}, ensure_ascii=False), flush=True)

    # The Generator is not loaded or forwarded before this point.  Only the
    # selected winner and the empirical S0 control now enter the canary.
    robust._prepare_cuda_runtime()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "step3000 Generator canary requires CUDA")
    device = torch.device("cuda:0")
    consumer, smooth, independent, ec = readiness._load_models(root, inputs["robust"], device)
    consumer.load_trainable_state_dict({"encoder.phase": inputs["checkpoint"]["encoder_state"], "generator.phase": inputs["checkpoint"]["generator_state"]})
    consumer.runtime.eval()
    for parameter in consumer.runtime.parameters():
        parameter.requires_grad_(False)
    models = {"E_S": smooth, "E_I": independent, "E_C": ec}
    for name, model in models.items():
        model.eval()
        _require(all(not parameter.requires_grad for parameter in model.parameters()), f"{name} is not frozen")

    control_z = train_z.index_select(0, torch.cat([split["validation_b_positions_by_class"][digit][:METRIC_PER_CLASS] for digit in range(CLASS_COUNT)]))
    winner_output = _run_canary_bank(consumer, models, winner_z, device, panel_indices=metric_indices)
    s0_output = _run_canary_bank(consumer, models, s0_z, device, panel_indices=metric_indices)
    control_output = _run_canary_bank(consumer, models, control_z, device, panel_indices=torch.arange(BANK_COUNT if control_z.shape[0] == BANK_COUNT else control_z.shape[0]))
    winner_semantic = _canary_semantic_report(winner_output, requested)
    s0_semantic = _canary_semantic_report(s0_output, requested)
    winner_canary = {
        "requested_vs_ols_probe": {
            "macro_recall": winner_report["probe"]["macro_recall"],
            "per_class_recall": winner_report["probe"]["per_class_recall"],
        },
        "semantic": winner_semantic,
        "retention": {name: float((candidate_internal[winner]["probe_predictions"] == winner_output["logits"][name].argmax(dim=1)).to(torch.float64).mean()) for name in models},
        "R_raw_R_feature": _canary_comparison(winner_output, control_output),
        "diversity": _canary_diversity(winner_z, winner_output),
        "optical_validity": {
            "finite": bool(winner_output["raw_stats"]["finite"]),
            "nonconstant_fraction": float(winner_output["raw_stats"]["nonconstant_fraction"]),
            "passes": bool(winner_output["raw_stats"]["finite"] and winner_output["raw_stats"]["nonconstant_sample_count"] == BANK_COUNT),
        },
        "raw_cmos": winner_output["raw_stats"],
    }
    s0_canary = {
        "requested_vs_ols_probe": {
            "macro_recall": s0_report["probe"]["macro_recall"],
            "per_class_recall": s0_report["probe"]["per_class_recall"],
        },
        "semantic": s0_semantic,
        "retention": {name: float((candidate_internal["S0"]["probe_predictions"] == s0_output["logits"][name].argmax(dim=1)).to(torch.float64).mean()) for name in models},
        "R_raw_R_feature": _canary_comparison(s0_output, control_output),
        "diversity": _canary_diversity(s0_z, s0_output),
        "optical_validity": {
            "finite": bool(s0_output["raw_stats"]["finite"]),
            "nonconstant_fraction": float(s0_output["raw_stats"]["nonconstant_fraction"]),
            "passes": bool(s0_output["raw_stats"]["finite"] and s0_output["raw_stats"]["nonconstant_sample_count"] == BANK_COUNT),
        },
        "raw_cmos": s0_output["raw_stats"],
    }
    classification, mechanism, successor = _classification(winner, winner_report, s0_report, candidate_reports, winner_canary, s0_canary)

    target.mkdir(parents=True, exist_ok=False)
    model_payload = {
        "schema": "semantically_conditioned_ols_sampler_models.v1",
        "split_identity": split_identity,
        "S1": {"family": "S1", "neighbor_count": S1_NEIGHBOR_COUNT, "standardization": {"mean": fit_mean, "std": fit_std}, "sampling": s1_sampling},
        "S2": {"family": "S2", "standardization": {"mean": fit_mean, "std": fit_std}, "model": s2_model},
        "S3": {"family": "S3", "standardization": {"mean": s3_mean, "std": s3_std}, "model": s3_model, "scale": scale},
    }
    bank_payload = {
        "schema": "semantically_conditioned_ols_sampler_candidate_banks.v1",
        "requested_labels": requested,
        "sample_indices": sample_indices,
        "paired_seeds": seeds,
        "z_by_family": candidate_z,
        "u_by_family": {name: statistical_variable(value, scale).to(torch.float64).cpu() for name, value in candidate_z.items()},
        "sample_contract": "sample(y, seed_or_index) with requested class and sample index persisted; 200 rows per class",
    }
    winner_payload = {
        "schema": "semantically_conditioned_ols_sampler_frozen_winner.v1",
        "family": winner,
        "selection": selection,
        "model_summary": {"S1": {"family": "S1", "neighbor_count": S1_NEIGHBOR_COUNT}, "S2": _model_summary(s2_model), "S3": _model_summary(s3_model)}[winner],
        "requested_labels": requested,
        "sample_indices": sample_indices,
        "paired_seeds": seeds,
        "z": winner_z,
        "u": statistical_variable(winner_z, scale).to(torch.float64).cpu(),
        "frozen_before_generator_canary": True,
    }
    panel_payload = {
        "schema": "semantically_conditioned_ols_sampler_generator_canary_metric_panels.v1",
        "panel_indices": metric_indices,
        "winner_raw_panel": winner_output["raw_panel"],
        "S0_raw_panel": s0_output["raw_panel"],
        "real_validation_control_raw_panel": control_output["raw_panel"],
        "control_role": "fixed real TRAIN validation B panel; labels not consumed by Generator or evaluator metrics",
    }
    _write_torch(target / "sampler_models.pt", model_payload, kind="semantic-conditioned sampler model parameters")
    _write_torch(target / "candidate_latent_banks.pt", bank_payload, kind="semantic-conditioned candidate latent banks")
    _write_torch(target / "winner_frozen_sampler.pt", winner_payload, kind="frozen selected semantic-conditioned sampler")
    _write_torch(target / "generator_canary_metric_panels.pt", panel_payload, kind="frozen Generator canary metric panels")

    source_files = [Path(__file__), root / "tests/v2/test_lightgen_2d_semantically_conditioned_ols_sampler_v1.py"]
    source_refs = [_artifact_ref(root, path.resolve(), "experiment source") for path in source_files if path.resolve().is_file()]
    input_authorities = {
        "real_train_complex_ols": _artifact_ref(root, root / TRAIN_OLS, "accepted real TRAIN complex OLS [55000,100]"),
        "real_dev_complex_ols": _artifact_ref(root, root / DEV_OLS, "accepted real DEV complex OLS [5000,100]"),
        "train_labels_and_split": {"source": inputs["robust"]["data_identity"], "labels_tensor_sha256": tensor_sha256(inputs["labels"])},
        "frozen_ols_probe": _artifact_ref(root, root / PROBE_WEIGHTS, "accepted frozen OLS linear probe"),
        "ols_scale_gauge_transport": {"scale_tensor_sha256": tensor_sha256(scale), "eps_s": EPS_S, "max_q_gauge": "accepted max_q_gauge; max-q with lowest-index tie", "transport": "T=I_100 coefficient space", "port_order": "row-major j=10*r+c", "writer": "W_OLS(z)=sum_j (Tz)_j v_j"},
        "accepted_p2_q2_bank": {**_artifact_ref(root, inputs["p2_path"], "fixed P2/Q2 sampled bank"), "requested_labels_available": False, "role": "descriptive legacy gap control only"},
        "accepted_manifold_gap": _artifact_ref(root, inputs["gap_path"], "accepted sampled OLS semantic-manifold gap diagnostic"),
        "frozen_step3000_generator": _artifact_ref(root, root / readiness.REAL_CHECKPOINTS[3000]["path"], "frozen step3000 Generator checkpoint"),
        "frozen_evaluators": inputs["preflight"]["preflight"]["frozen_evaluators"],
    }
    report = {
        "schema": "semantically_conditioned_ols_sampler_v1_report",
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "task_id": TASK_ID,
        "mode": "IMPLEMENT_PREPARED",
        "risk_level": "R2_ARTIFACT_PRODUCING_RESEARCH",
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "input_authority": INPUT_GATE,
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": True,
        "local_state": {"branch": inputs["git"]["branch"], "head": inputs["git"]["head"], "worktree_dirty": bool(inputs["git"].get("worktree_dirty", True)), "working_tree_preserved": True},
        "supported_bottleneck": SUPPORTED_BOTTLENECK,
        "sampler_target": "p(z|y), z in C^100, y in {0,...,9}",
        "train_validation_roles": {"SAMPLER_FIT": split_identity, "SAMPLER_VALIDATION": {"count": 5000, "per_class": 500, "split_seed": SPLIT_SEED}, "DEV": "read-only authority reproduction and final contextual diagnostic; not fit/rank/covariance/selection/tie-break"},
        "candidate_domain": ["S0", "S1", "S2", "S3"],
        "primary_representation": "original complex OLS z in C^100; Cartesian and gauged amplitude-phase forms are internal model coordinates only",
        "primary_selection_statistic": "macro semantic_support_rate over requested classes; success iff frozen OLS probe top-1=requested y AND same-class SAMPLER_FIT_y distance<=SAMPLER_VALIDATION class-y q95",
        "secondary_statistics": ["probe confidence", "top1-top2 margin", "per-class support", "nearest-real distance/class agreement", "real-class centroid/logit geometry", "R_u", "R_z", "amplitude/phase compatibility", "balance", "finite/nonconstant"],
        "local_support_contract": {"distance": "Euclidean distance in global SAMPLER_FIT-standardized [Re(z),Im(z)] R^200", "threshold": "per-class q95 of SAMPLER_VALIDATION-to-SAMPLER_FIT_y distances", "posthoc_dev_rejection": False},
        "class_balance_contract": {"requested_bank_rows": BANK_COUNT, "rows_per_class": BANK_PER_CLASS, "requested_labels_persisted": True, "paired_sample_indices": True},
        "complex_gauge_contract": {"returned_coordinate": "complex64 [N,100] in accepted Generator-compatible OLS ordering", "S1_S2": "no phase/gauge manipulation beyond Cartesian inverse reconstruction", "S3": "u=|z|^2/(s+eps), max-q gauge, wrapped phase pair normalization, accepted max-q reconstruction", "finite_required": True},
        "S0_control_contract": "deterministic no-replacement exact rows from SAMPLER_FIT_y; empirical-support control and not eligible as generative winner",
        "S1_contract": "same-class 16NN local chord interpolation in global TRAIN-FIT standardized Cartesian OLS; per-row deterministic anchor, deterministic neighbor rank, Uniform(0,1) lambda",
        "S2_contract": "class-conditioned joint Cartesian low-rank Gaussian; smallest rank explaining 95% within-class variance with cap32 and fixed isotropic residual floor",
        "S3_contract": "class-conditioned joint [log(u),cos(phi_gauge),sin(phi_gauge)] low-rank Gaussian; same 95%/cap32/residual rule and exact complex reconstruction",
        "complexity_treatment": {"pca_variance_fraction": PCA_VARIANCE_FRACTION, "rank_cap": PCA_RANK_CAP, "residual_variance_floor": RESIDUAL_VARIANCE_FLOOR, "grid_search": False, "rejection_sampling": False},
        "tie_break": selection,
        "one_shot_selection_semantics": "one winner selected from S1-S3 in OLS latent space; Generator/evaluators/R_z do not alter winner",
        "post_selection_generator_canary": {"checkpoint_step": 3000, "banks": [winner, "S0"], "updates": False, "backward": False, "optimizer": False, "requested_class_is_ground_truth": True, "real_validation_control_used_for_R_metrics": True},
        "latent_candidate_results": {"S0": candidate_reports["S0"], "S1": candidate_reports["S1"], "S2": candidate_reports["S2"], "S3": candidate_reports["S3"]},
        "winner": {"family": winner, "frozen": True, "selection": selection, "absolute_target": {"target": LATENT_TARGET, "winner_rate": winner_report["semantic_support_rate"], "winner_ci_lower": winner_report["semantic_support_rate_bootstrap_ci"]["lower"], "passes": bool(float(winner_report["semantic_support_rate"]) >= LATENT_TARGET and float(winner_report["semantic_support_rate_bootstrap_ci"]["lower"]) >= LATENT_TARGET)}},
        "legacy_p2_q2_context": {"classification": SUPPORTED_BOTTLENECK, "old_predicted_class_real_dev_q95_support_fraction": OLD_Q2_PREDCLASS_SUPPORT, "requested_labels_consumed": False},
        "winner_canary": winner_canary,
        "S0_control_canary": s0_canary,
        "causal_comparison": {"winner_vs_S0_E_C_macro_recall_delta": float(winner_canary["semantic"]["E_C"]["macro_recall"] - s0_canary["semantic"]["E_C"]["macro_recall"]), "winner_vs_S0_R_raw_delta": float(winner_canary["R_raw_R_feature"]["R_raw"] - s0_canary["R_raw_R_feature"]["R_raw"]), "winner_vs_S0_R_feature_delta": float(winner_canary["R_raw_R_feature"]["R_feature"] - s0_canary["R_raw_R_feature"]["R_feature"])},
        "classification": classification,
        "mechanism": mechanism,
        "recommended_single_successor": successor,
        "success_failure_classifications": ["SEMANTICALLY_CONDITIONED_OLS_SAMPLER_SUPPORTED", "SEMANTIC_OLS_SUPPORT_REPAIRED_BUT_GENERATOR_TRANSFER_MIXED", "EMPIRICAL_REAL_OLS_SUPPORT_WORKS_BUT_GENERATIVE_MODELING_GAP_REMAINS", "REAL_OLS_REPLAY_DOES_NOT_RECOVER_EXPECTED_GENERATOR_SEMANTICS", "SEMANTICALLY_CONDITIONED_OLS_SAMPLER_NOT_SUPPORTED", "SEMANTICALLY_CONDITIONED_OLS_SAMPLER_MIXED"],
        "input_authorities": input_authorities,
        "scientific_artifacts": {name: _artifact_ref(root, target / name, role) for name, role in (("sampler_models.pt", "fitted frozen statistical sampler parameters"), ("candidate_latent_banks.pt", "balanced S0-S3 latent banks"), ("winner_frozen_sampler.pt", "frozen selected sampler"), ("generator_canary_metric_panels.pt", "winner/S0/control raw metric panels"))},
        "files_changed": {"repository_source": [_relative(root, Path(__file__).resolve()), _relative(root, root / "tests/v2/test_lightgen_2d_semantically_conditioned_ols_sampler_v1.py")], "status_documents_modified": False},
        "source_refs": source_refs,
        "validation": {"authority_preflight": "READY", "cuda": True, "forward_only_generator_canary": True, "backward_called": False, "optimizer_constructed": False, "neural_training": False, "statistical_sampler_fitting": True, "generator_forward_before_winner_selection": False, "sampler_selection_dev_used": False, "protected_test_sealed_reads": False, "requested_class_identity_persisted": True, "old_p2_q2_requested_labels_consumed": False},
        "claim_boundary": "This artifact establishes one bounded class-conditioned OLS sampler comparison and one post-selection step3000 Generator canary. It does not authorize optical training, architecture changes, promotion, or final generation quality claims.",
        "unresolved": ["Downstream canary results are limited to the frozen step3000 Generator and the named frozen evaluators; no alternate sampler may be selected after seeing them."],
        "next_action": "NONE",
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report, kind="semantic-conditioned OLS sampler report")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the explicit authorization and execute the frozen comparison")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[3]
    if not args.execute:
        target = (root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)).resolve()
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "output_dir": _relative(root, target), "no_update": True}, ensure_ascii=False, indent=2))
        return 0
    report = _run_experiment(root, Path(args.output_dir))
    print(json.dumps({"result": report["result"], "outer_gate": report["outer_gate"], "classification": report["classification"], "winner": report["winner"]["family"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
