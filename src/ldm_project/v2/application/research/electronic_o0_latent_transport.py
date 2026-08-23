"""No-optics attribution and latent-only transport for the corrected WAE O0.

The WAE encoder and decoder are immutable inputs.  The only trainable object
in the second phase is a small conditional eight-dimensional RealNVP that
maps a standard Gaussian code to the frozen encoder posterior.  Evaluators
are acceptance instruments and are never used by attribution, latent fitting,
or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ...evidence import (
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ...student.conditional import camera_code_fraction, relative_diversity_ratios
from ...student.conditional_acceptance import conditional_coverage_ratio
from ...student.evaluator import (
    SmoothMNISTEvaluator,
    load_frozen_evaluator,
    load_local_mnist,
    stratified_mnist_indices,
)
from .electronic_o0_wae import (
    LOCKED_SEEDS,
    _bootstrap_ci,
    _eval_model,
    _memorization_audit,
    _recall,
    balanced_epoch_schedule,
    panel_codes,
)
from .wae_models import ConditionalWAEEncoder, ConditionalWAEDecoder


SCHEMA_PREFIX = "electronic_o0_latent_transport"
FLOW_SEEDS = (8401, 8402, 8403)
BOOTSTRAP_SEED = 90210
CODE_DIM = 8
OUTPUT_ROOT = Path(
    "outputs/20260823_v2/runs/20260823_electronic_o0_latent_transport_v1"
)


def _git() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "clean": not status, "status": status}


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _write_checked(path: str | Path, payload: dict[str, Any]) -> None:
    candidate = Path(path)
    run_root = candidate.parent if candidate.parent.name.startswith("20260823_") else candidate.parents[1]
    if run_root.exists() and not _storage_ok(run_root):
        raise RuntimeError("latent transport storage gate failed before evidence write")
    _write(path, payload)
    if not _storage_ok(run_root):
        raise RuntimeError("latent transport storage gate failed after evidence write")


def _config(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("code_dim") != CODE_DIM
        or tuple(payload.get("flow_seeds", ())) != FLOW_SEEDS
        or tuple(payload.get("locked_panel_seeds", ())) != LOCKED_SEEDS
        or payload.get("harness_revision") != 1
    ):
        raise ValueError("latent transport config is not the registered v1 contract")
    return payload


def _require_files(paths: list[str | Path], dataset_root: str | Path) -> None:
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("latent transport lineage is missing: " + ", ".join(missing))
    if not Path(dataset_root).is_dir():
        raise FileNotFoundError(f"MNIST dataset root is missing: {dataset_root}")


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    paths = [
        args.config,
        args.wae_training_checkpoint,
        args.wae_decoder_checkpoint,
        args.wae_confirm_evidence,
        args.primary_evaluator_checkpoint,
        args.secondary_evaluator_checkpoint,
        args.primary_qualification_evidence,
        args.secondary_qualification_evidence,
    ]
    _require_files(paths, args.dataset_root)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("formal latent transport lifecycle requires CUDA")
    if args.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUDA lifecycle requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    return {
        "git": _git(),
        "config_sha256": file_sha256(args.config),
        "config_content": config,
        "wae_training_sha256": file_sha256(args.wae_training_checkpoint),
        "wae_decoder_sha256": file_sha256(args.wae_decoder_checkpoint),
        "wae_confirm_sha256": file_sha256(args.wae_confirm_evidence),
        "primary_evaluator_sha256": file_sha256(args.primary_evaluator_checkpoint),
        "secondary_evaluator_sha256": file_sha256(args.secondary_evaluator_checkpoint),
        "primary_qualification_sha256": file_sha256(args.primary_qualification_evidence),
        "secondary_qualification_sha256": file_sha256(args.secondary_qualification_evidence),
        "dataset_root": str(args.dataset_root),
        "device": args.device,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "seeds": {"flow": list(FLOW_SEEDS), "locked": list(LOCKED_SEEDS), "bootstrap": BOOTSTRAP_SEED},
    }


def _load_wae(
    training_path: str | Path, decoder_path: str | Path, device: torch.device
) -> tuple[ConditionalWAEEncoder, ConditionalWAEDecoder]:
    training = safe_torch_load(training_path)
    if not isinstance(training, dict) or training.get("schema") != "electronic_o0_wae_training.v2":
        raise ValueError("latent transport requires an electronic WAE v2 training checkpoint")
    decoder_payload = safe_torch_load(decoder_path)
    if (
        not isinstance(decoder_payload, dict)
        or decoder_payload.get("schema") != "electronic_o0_wae_decoder.v2"
        or any(key in decoder_payload for key in ("encoder", "optimizer"))
    ):
        raise ValueError("latent transport decoder checkpoint is not a decoder-only WAE v2 artifact")
    encoder = ConditionalWAEEncoder().to(device)
    decoder = ConditionalWAEDecoder("O0-I").to(device)
    encoder.load_state_dict(training["encoder"], strict=True)
    decoder.load_state_dict(decoder_payload["decoder_state_dict"], strict=True)
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def _encode(
    encoder: ConditionalWAEEncoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            values.append(
                encoder(
                    images[start : start + batch_size].to(device),
                    labels[start : start + batch_size].to(device),
                ).cpu()
            )
    return torch.cat(values)


def _posterior(
    encoder: ConditionalWAEEncoder,
    dataset_root: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    images, labels = load_local_mnist(dataset_root, train=True)
    train_index, validation_index = stratified_mnist_indices(labels)
    train_images, train_labels = images.index_select(0, train_index), labels.index_select(0, train_index)
    validation_images, validation_labels = images.index_select(0, validation_index), labels.index_select(0, validation_index)
    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "validation_images": validation_images,
        "validation_labels": validation_labels,
        "train_codes": _encode(encoder, train_images, train_labels, device),
        "validation_codes": _encode(encoder, validation_images, validation_labels, device),
    }


def _stats(codes: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    means, cholesky, covariances = [], [], []
    for label in range(10):
        value = codes[labels == label]
        mean = value.mean(0)
        covariance = torch.cov(value.T) + 1.0e-4 * torch.eye(CODE_DIM)
        means.append(mean)
        covariances.append(covariance)
        cholesky.append(torch.linalg.cholesky(covariance))
    global_mean = codes.mean(0)
    global_covariance = torch.cov(codes.T) + 1.0e-4 * torch.eye(CODE_DIM)
    return {
        "means": torch.stack(means),
        "covariances": torch.stack(covariances),
        "cholesky": torch.stack(cholesky),
        "global_mean": global_mean,
        "global_cholesky": torch.linalg.cholesky(global_covariance),
    }


def _transport(
    epsilon: torch.Tensor,
    labels: torch.Tensor,
    statistics: dict[str, Any],
    branch: str,
    empirical: torch.Tensor | None = None,
) -> torch.Tensor:
    if branch == "P0":
        return epsilon
    if branch == "Aglobal":
        return statistics["global_mean"].to(epsilon.device) + epsilon @ statistics["global_cholesky"].to(epsilon.device).T
    if branch == "Aclass":
        mean = statistics["means"].to(epsilon.device).index_select(0, labels)
        chol = statistics["cholesky"].to(epsilon.device).index_select(0, labels)
        return mean + torch.bmm(chol, epsilon.unsqueeze(-1)).squeeze(-1)
    if branch in {"Eposterior", "Rencoded"} and empirical is not None:
        return empirical.to(epsilon.device)
    raise ValueError(f"unknown latent transport branch {branch}")


def _sliced_distance(left: torch.Tensor, right: torch.Tensor, *, seed: int = 1234, directions: int = 64) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn((left.shape[1], directions), generator=generator)
    projection = F.normalize(projection, dim=0)
    left_values = torch.sort(left.cpu() @ projection, dim=0).values
    right_values = torch.sort(right.cpu() @ projection, dim=0).values
    count = min(left_values.shape[0], right_values.shape[0])
    return float((left_values[:count] - right_values[:count]).abs().mean())


def _latent_summary(generated: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    return {
        "mean_error": float((generated.mean(0) - target.mean(0)).norm()),
        "covariance_error": float((torch.cov(generated.T) - torch.cov(target.T)).norm()),
        "sliced_distance": _sliced_distance(generated, target),
        "norm_mean": float(generated.norm(dim=1).mean()),
        "norm_p95": float(torch.quantile(generated.norm(dim=1), 0.95)),
        "class_mean_norms": [float(generated[labels == label].mean(0).norm()) for label in range(10)],
        "rank": int(torch.linalg.matrix_rank(torch.cov(generated.T), tol=1.0e-3)),
    }


def _balanced_empirical(codes: torch.Tensor, labels: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for label in range(10):
        choices = torch.nonzero(labels == label, as_tuple=False).flatten()
        rows.append(choices.index_select(0, torch.randperm(choices.numel(), generator=generator)[:50]))
    return codes.index_select(0, torch.cat(rows))


def _decode_views(
    decoder: ConditionalWAEDecoder,
    codes: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = 100,
) -> torch.Tensor:
    """Decode with the WAE measurement batch contract and return CPU views."""
    views = []
    with torch.no_grad():
        for start in range(0, codes.shape[0], batch_size):
            views.append(decoder(codes[start:start + batch_size], labels[start:start + batch_size]).cpu())
    return torch.cat(views, dim=0)


def _evaluate_views(
    model: nn.Module,
    views: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a frozen evaluator on CPU-held views without changing its device."""
    return _eval_model(model, views.to(device))


def _metrics(
    decoder: ConditionalWAEDecoder,
    z: torch.Tensor,
    labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    primary: nn.Module,
    secondary: nn.Module,
    *,
    branch: str,
) -> dict[str, Any]:
    views = _decode_views(decoder, z, labels)
    reference_index = torch.cat([
        torch.nonzero(test_labels == label, as_tuple=False).flatten()[:50]
        for label in range(10)
    ])
    reference_images = test_images.index_select(0, reference_index)
    device = next(primary.parameters()).device
    p_features, p_logits = _evaluate_views(primary, views, device)
    s_features, s_logits = _evaluate_views(secondary, views, device)
    p_ref, _ = _evaluate_views(primary, reference_images, device)
    s_ref, _ = _evaluate_views(secondary, reference_images, device)
    p_pred, s_pred = p_logits.argmax(1), s_logits.argmax(1)
    p_recall, s_recall = _recall(p_pred, labels.cpu()), _recall(s_pred, labels.cpu())
    shuffled = labels.cpu().index_select(
        0, torch.randperm(labels.numel(), generator=torch.Generator(device="cpu").manual_seed(8675309))
    ).to(labels.device)
    shuffled_views = _decode_views(decoder, z, shuffled)
    shuffled_prediction = _evaluate_views(primary, shuffled_views, device)[1].argmax(1)
    coverage = {
        "primary": conditional_coverage_ratio(p_features, views, p_ref, reference_images, labels.cpu(), epsilon=0.05, iterations=20),
        "secondary": conditional_coverage_ratio(s_features, views, s_ref, reference_images, labels.cpu(), epsilon=0.05, iterations=20),
    }
    diversity = {
        "primary": relative_diversity_ratios(p_features, p_ref, labels.cpu()),
        "secondary": relative_diversity_ratios(s_features, s_ref, labels.cpu()),
    }
    return {
        "branch": branch,
        "primary": {"accuracy": float((p_pred == labels.cpu()).float().mean()), "recall": p_recall, "minimum_recall": min(p_recall)},
        "secondary": {"accuracy": float((s_pred == labels.cpu()).float().mean()), "recall": s_recall, "minimum_recall": min(s_recall)},
        "agreement": float((p_pred == s_pred).float().mean()),
        "camera_code_fraction": float(camera_code_fraction(views, labels.cpu())),
        "saturation": float((views >= 1.0).float().mean()),
        "coverage": coverage,
        "diversity_ratio": diversity,
        "label_shuffled_primary_accuracy": float((shuffled_prediction == labels.cpu()).float().mean()),
        "prediction_histogram": {"primary": torch.bincount(p_pred, minlength=10).tolist(), "secondary": torch.bincount(s_pred, minlength=10).tolist()},
        "view_hash": tensor_sha256(views),
        "bootstrap_accuracy_ci": {"primary": _bootstrap_ci((p_pred == labels.cpu()).float(), seed=BOOTSTRAP_SEED), "secondary": _bootstrap_ci((s_pred == labels.cpu()).float(), seed=BOOTSTRAP_SEED + 1)},
    }


def _dual_passed(row: dict[str, Any]) -> bool:
    return (
        row["primary"]["accuracy"] >= 0.95
        and row["secondary"]["accuracy"] >= 0.95
        and row["primary"]["minimum_recall"] >= 0.90
        and row["secondary"]["minimum_recall"] >= 0.90
        and row["agreement"] >= 0.95
        and row["camera_code_fraction"] >= 0.25
        and row["saturation"] <= 0.08
        and row["label_shuffled_primary_accuracy"] <= 0.20
        and all(0.70 <= value <= 1.30 for values in row["diversity_ratio"].values() for value in values)
        and all(value <= 1.50 for value in row["coverage"].values())
    )


def _load_evaluators(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module]:
    primary, _ = load_frozen_evaluator(args.primary_evaluator_checkpoint, device=device)
    secondary_payload = safe_torch_load(args.secondary_evaluator_checkpoint)
    if not isinstance(secondary_payload, dict) or secondary_payload.get("architecture_kind") != SmoothMNISTEvaluator.architecture_kind:
        raise ValueError("secondary evaluator qualification is invalid")
    secondary = SmoothMNISTEvaluator().to(device)
    secondary.load_state_dict(secondary_payload["state_dict"], strict=True)
    secondary.freeze()
    return primary, secondary


def _validate_upstream(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(Path(args.wae_confirm_evidence).read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "electronic_o0_wae_confirm.v2"
        or payload.get("status") != "valid"
        or payload.get("classification") != "wae_image_manifold_failed"
    ):
        raise ValueError("latent transport requires the corrected valid WAE v2 failure")
    if payload.get("checkpoint") != str(args.wae_decoder_checkpoint):
        raise ValueError("WAE confirm decoder path does not match latent transport lineage")
    training_path = payload.get("training", {}).get("training_checkpoint")
    if training_path != str(args.wae_training_checkpoint) or file_sha256(training_path) != file_sha256(args.wae_training_checkpoint):
        raise ValueError("WAE confirm training checkpoint does not match latent transport lineage")
    return payload


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[name], right[name]) for name in left)


def _optimizer_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["param_groups"] != right["param_groups"] or left["state"].keys() != right["state"].keys():
        return False
    for key in left["state"]:
        for name, value in left["state"][key].items():
            other = right["state"][key][name]
            if isinstance(value, torch.Tensor):
                if not torch.equal(value, other):
                    return False
            elif value != other:
                return False
    return True


class AffineCoupling(nn.Module):
    def __init__(self, mask: torch.Tensor, hidden: int = 128) -> None:
        super().__init__()
        self.register_buffer("mask", mask.to(torch.bool))
        input_dim = int(self.mask.sum()) + 32
        output_dim = int((~self.mask).sum()) * 2
        self.label = nn.Embedding(10, 32)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, output_dim))

    def forward(self, values: torch.Tensor, labels: torch.Tensor, *, inverse: bool) -> tuple[torch.Tensor, torch.Tensor]:
        kept = values[:, self.mask]
        updated = values[:, ~self.mask]
        raw = self.net(torch.cat((kept, self.label(labels)), dim=1))
        scale, shift = raw.chunk(2, dim=1)
        scale = 2.0 * torch.tanh(scale)
        if inverse:
            updated = (updated - shift) * torch.exp(-scale)
            logdet = -scale.sum(1)
        else:
            updated = updated * torch.exp(scale) + shift
            logdet = scale.sum(1)
        result = values.clone()
        result[:, ~self.mask] = updated
        return result, logdet


class LatentTransportFlow(nn.Module):
    def __init__(self, blocks: int = 8, hidden: int = 128) -> None:
        super().__init__()
        if blocks != 8 or hidden != 128:
            raise ValueError("latent flow architecture is fixed to 8 blocks and hidden=128")
        self.blocks = nn.ModuleList([
            AffineCoupling(torch.tensor([(index + block) % 2 == 0 for index in range(CODE_DIM)]), hidden)
            for block in range(blocks)
        ])
        permutations = []
        for block in range(blocks - 1):
            permutation = torch.arange(CODE_DIM)
            if block % 2 == 0:
                permutation = permutation.flip(0)
            permutations.append(permutation)
        self.register_buffer("permutations", torch.stack(permutations))

    def forward(self, codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        values = codes
        for index, block in enumerate(self.blocks):
            values, _ = block(values, labels, inverse=False)
            if index < len(self.blocks) - 1:
                values = values[:, self.permutations[index]]
        return values

    def inverse(self, values: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        result = values
        logdet = values.new_zeros(values.shape[0])
        for index in reversed(range(len(self.blocks))):
            if index < len(self.blocks) - 1:
                permutation = self.permutations[index]
                inverse_permutation = torch.argsort(permutation)
                result = result[:, inverse_permutation]
            result, change = self.blocks[index](result, labels, inverse=True)
            logdet = logdet + change
        return result, logdet


def _flow_nll(flow: LatentTransportFlow, codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    base, logdet = flow.inverse(codes, labels)
    constant = 0.5 * CODE_DIM * math.log(2.0 * math.pi)
    return (0.5 * base.square().sum(1) + constant - logdet).mean()


def _flow_update(flow: LatentTransportFlow, optimizer: torch.optim.Optimizer, codes: torch.Tensor, labels: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = _flow_nll(flow, codes, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("latent flow loss became non-finite")
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _flow_train(
    *,
    train_codes: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
    seed: int,
    epochs: int,
    evidence_path: str | Path,
    resume: str | None,
) -> tuple[Path, Path, dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    flow = LatentTransportFlow().to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=1.0e-3)
    start_epoch, step = 0, 0
    if resume:
        payload = safe_torch_load(resume)
        if not isinstance(payload, dict) or payload.get("schema") != "electronic_o0_latent_transport_training.v1":
            raise ValueError("latent flow resume checkpoint schema mismatch")
        flow.load_state_dict(payload["flow"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch, step = int(payload["epoch"]), int(payload["step"])
        rng = payload.get("rng_state", {})
        torch.set_rng_state(rng["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(rng["cuda"])
    labels_cpu = train_labels.cpu()
    losses = []
    for epoch in range(start_epoch, epochs):
        schedule = balanced_epoch_schedule(labels_cpu, seed=seed + epoch)
        for index in schedule:
            losses.append(_flow_update(flow, optimizer, train_codes.index_select(0, index).to(device), labels_cpu.index_select(0, index).to(device)))
            step += 1
    training_path = Path(evidence_path).with_name(f"seed_{seed}_training.pt")
    atomic_torch_save(training_path, {
        "schema": "electronic_o0_latent_transport_training.v1",
        "seed": seed,
        "epoch": epochs,
        "step": step,
        "flow": flow.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []},
        "schedule": {"steps": len(schedule), "unique_samples": int(torch.unique(torch.cat(schedule)).numel())},
    })
    inference_path = Path(evidence_path).with_name(f"seed_{seed}_flow.pt")
    atomic_torch_save(inference_path, {
        "schema": "electronic_o0_latent_transport_flow.v1",
        "seed": seed,
        "epoch": epochs,
        "flow": flow.state_dict(),
        "parameter_count": sum(parameter.numel() for parameter in flow.parameters()),
        "source_training_sha256": file_sha256(training_path),
    })
    return inference_path, training_path, {"seed": seed, "epoch": epochs, "step": step, "loss_final": losses[-1] if losses else None, "schedule": {"steps": len(schedule), "unique_samples": int(torch.unique(torch.cat(schedule)).numel())}}


def _load_flow(path: str | Path, device: torch.device) -> LatentTransportFlow:
    payload = safe_torch_load(path)
    if not isinstance(payload, dict) or payload.get("schema") != "electronic_o0_latent_transport_flow.v1" or any(key in payload for key in ("encoder", "optimizer")):
        raise ValueError("latent flow inference artifact is invalid")
    flow = LatentTransportFlow().to(device)
    flow.load_state_dict(payload["flow"], strict=True)
    flow.eval()
    return flow


def _flow_measure(
    flow_path: str | Path,
    decoder: ConditionalWAEDecoder,
    device: torch.device,
    dataset_root: str,
    primary_path: str,
    secondary_path: str,
) -> dict[str, Any]:
    flow = _load_flow(flow_path, device)
    primary, secondary = _load_evaluators(argparse.Namespace(primary_evaluator_checkpoint=primary_path, secondary_evaluator_checkpoint=secondary_path), device)
    test_images, test_labels = load_local_mnist(dataset_root, train=False)
    all_images, all_labels = load_local_mnist(dataset_root, train=True)
    train_index, validation_index = stratified_mnist_indices(all_labels)
    train_images, train_labels = all_images.index_select(0, train_index), all_labels.index_select(0, train_index)
    validation_images, validation_labels = all_images.index_select(0, validation_index), all_labels.index_select(0, validation_index)
    rows = {}
    for panel_seed in LOCKED_SEEDS:
        codes, labels = panel_codes(panel_seed, device=device)
        with torch.no_grad():
            z = flow(codes, labels)
        row = _metrics(decoder, z, labels, test_images, test_labels, primary, secondary, branch="flow")
        row["latent_norm_mean"] = float(z.norm(dim=1).mean())
        row["latent_norm_p95"] = float(torch.quantile(z.norm(dim=1), 0.95))
        row["no_update_replay_exact"] = tensor_sha256(z.cpu()) == tensor_sha256(flow(codes, labels).cpu())
        rows[str(panel_seed)] = row
    official_labels = test_labels.to(device)
    official_codes = torch.randn((official_labels.numel(), CODE_DIM), generator=torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)).to(device)
    with torch.no_grad():
        official_z = flow(official_codes, official_labels)
    official_views = _decode_views(decoder, official_z, official_labels)
    evaluator_device = next(primary.parameters()).device
    official_p = _evaluate_views(primary, official_views, evaluator_device)[1].argmax(1)
    official_s = _evaluate_views(secondary, official_views, evaluator_device)[1].argmax(1)
    official = {
        "count": int(official_labels.numel()),
        "primary_accuracy": float((official_p == official_labels.cpu()).float().mean()),
        "secondary_accuracy": float((official_s == official_labels.cpu()).float().mean()),
        "primary_minimum_recall": min(_recall(official_p, official_labels.cpu())),
        "secondary_minimum_recall": min(_recall(official_s, official_labels.cpu())),
        "agreement": float((official_p == official_s).float().mean()),
        "camera_code_fraction": float(camera_code_fraction(official_views, official_labels.cpu())),
        "saturation": float((official_views >= 1.0).float().mean()),
        "prediction_histogram": {"primary": torch.bincount(official_p, minlength=10).tolist(), "secondary": torch.bincount(official_s, minlength=10).tolist()},
    }
    jacobian_rows = []
    for panel_seed in LOCKED_SEEDS:
        codes, labels = panel_codes(panel_seed, device=device)
        for label in range(10):
            code = codes[label * 50].detach().requires_grad_(True)
            target = labels[label * 50].reshape(1)
            columns = []
            for direction in torch.eye(CODE_DIM, device=device):
                _, derivative = torch.autograd.functional.jvp(lambda value: decoder(flow(value.unsqueeze(0), target), target).flatten(), code, direction)
                columns.append(derivative)
            singular = torch.linalg.svdvals(torch.stack(columns, dim=1))
            jacobian_rows.append({"rank_1e-3": int((singular >= singular[0].clamp_min(1e-12) * 1e-3).sum()), "rank_1e-8": int((singular >= singular[0].clamp_min(1e-12) * 1e-8).sum()), "singular_values": [float(value) for value in singular.cpu()]})
    memory = _memorization_audit(official_views, official_labels.cpu(), train_images, train_labels, validation_images, validation_labels, device=device)
    gates = {
        "primary_accuracy": all(row["primary"]["accuracy"] >= 0.95 for row in rows.values()),
        "secondary_accuracy": all(row["secondary"]["accuracy"] >= 0.95 for row in rows.values()),
        "primary_minimum_recall": all(row["primary"]["minimum_recall"] >= 0.90 for row in rows.values()),
        "secondary_minimum_recall": all(row["secondary"]["minimum_recall"] >= 0.90 for row in rows.values()),
        "agreement": all(row["agreement"] >= 0.95 for row in rows.values()),
        "camera_code_fraction": all(row["camera_code_fraction"] >= 0.25 for row in rows.values()),
        "diversity": all(0.70 <= value <= 1.30 for row in rows.values() for values in row["diversity_ratio"].values() for value in values),
        "coverage": all(value <= 1.50 for row in rows.values() for value in row["coverage"].values()),
        "saturation": all(row["saturation"] <= 0.08 for row in rows.values()),
        "label_shuffled": all(row["label_shuffled_primary_accuracy"] <= 0.20 for row in rows.values()),
        "jacobian": sum(row["rank_1e-3"] >= 7 for row in jacobian_rows) / len(jacobian_rows) >= 0.80,
        "memorization": bool(memory["passed"]),
    }
    return {"flow_checkpoint": str(flow_path), "flow_checkpoint_sha256": file_sha256(flow_path), "panels": rows, "official_test": official, "jacobian": {"points": jacobian_rows, "rank_1e-3_pass_fraction": sum(row["rank_1e-3"] >= 7 for row in jacobian_rows) / len(jacobian_rows)}, "memorization": memory, "gates": gates, "passed": all(gates.values()), "evaluator_calls_during_training": 0}


def _storage_ok(root: Path) -> bool:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) <= 2 * 1024**3


def classify_attribution(payload: dict[str, Any]) -> tuple[str, str]:
    if payload.get("status") == "invalid_harness":
        return "invalid_harness", "repair_and_revalidate_on_a_new_clean_commit"
    branches = payload["branches"]
    if branches["Aclass"]["passed"]:
        return "affine_latent_transport_candidate", "register_class_conditional_affine_adapter"
    if branches["Eposterior"]["passed"] and branches["Rencoded"]["passed"]:
        return "nonlinear_latent_transport_candidate", "run_latent_flow_preflight"
    if not branches["Eposterior"]["passed"] or not branches["Rencoded"]["passed"]:
        return "decoder_posterior_manifold_failed", "close_current_electronic_and_fiber_route"
    return "latent_transport_inconclusive", "hold_no_training_request_new_hypothesis"


def dispatch(args: argparse.Namespace) -> int:
    config = _config(args.config)
    lineage = _lineage(args, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.research_action == "validate":
        upstream = _validate_upstream(args)
        if not lineage["git"]["clean"]:
            raise RuntimeError("latent transport validation requires a clean worktree")
        device = torch.device(args.device)
        encoder, decoder = _load_wae(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        count = sum(parameter.numel() for parameter in encoder.parameters()) + sum(parameter.numel() for parameter in decoder.parameters())
        payload = {"schema": f"{SCHEMA_PREFIX}_validation.v1", "status": "valid", "lineage": lineage, "upstream_classification": upstream["classification"], "wae_parameter_count": count, "flow_contract": {"code_dim": CODE_DIM, "blocks": 8, "hidden": 128}, "optimizer_created": False, "evaluator_calls": 0, "formal_promotion": False, "training_authorized": False, "synthetic_diagnostic_only": True}
        _write_checked(output, payload)
        return 0
    if args.research_action == "attribute":
        validation = json.loads(Path(args.validation_evidence).read_text(encoding="utf-8"))
        if validation.get("status") != "valid":
            raise ValueError("attribution requires valid latent transport validation")
        upstream = _validate_upstream(args)
        device = torch.device(args.device)
        encoder, decoder = _load_wae(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        posterior = _posterior(encoder, args.dataset_root, device)
        statistics = _stats(posterior["train_codes"], posterior["train_labels"])
        primary, secondary = _load_evaluators(args, device)
        test_images, test_labels = load_local_mnist(args.dataset_root, train=False)
        branches: dict[str, Any] = {}
        confirm_measurement = upstream["measurement"]["panels"]
        for branch in ("P0", "Aglobal", "Aclass", "Eposterior", "Rencoded"):
            panel_rows = {}
            latent_rows = {}
            for panel_seed in LOCKED_SEEDS:
                epsilon, labels = panel_codes(panel_seed, device=device)
                empirical = _balanced_empirical(posterior["validation_codes"], posterior["validation_labels"], panel_seed + 2718)
                selected = empirical if branch in {"Eposterior", "Rencoded"} else None
                z = _transport(epsilon, labels, statistics, branch, selected)
                panel_rows[str(panel_seed)] = _metrics(decoder, z, labels, test_images, test_labels, primary, secondary, branch=branch)
                latent_rows[str(panel_seed)] = _latent_summary(z.cpu(), posterior["validation_codes"], labels.cpu())
                if branch == "P0" and panel_rows[str(panel_seed)]["view_hash"] != confirm_measurement[str(panel_seed)]["view_hash"]:
                    raise RuntimeError("P0 latent transport does not reproduce WAE v2 confirm tensor")
            branches[branch] = {"panels": panel_rows, "latent": latent_rows, "passed": all(_dual_passed(row) for row in panel_rows.values()), "nonphysical": branch in {"Eposterior", "Rencoded"}}
        classification, next_action = classify_attribution({"branches": branches})
        payload = {"schema": f"{SCHEMA_PREFIX}_attribution.v1", "status": "valid", "lineage": lineage, "upstream_confirm_sha256": file_sha256(args.wae_confirm_evidence), "posterior": {"train_count": int(posterior["train_codes"].shape[0]), "validation_count": int(posterior["validation_codes"].shape[0]), "train_hash": tensor_sha256(posterior["train_codes"]), "validation_hash": tensor_sha256(posterior["validation_codes"])}, "branches": branches, "classification": classification, "next_action": next_action, "formal_promotion": False, "training_authorized": False, "synthetic_diagnostic_only": True}
        _write_checked(output, payload)
        return 0
    if args.research_action == "preflight":
        attribution = json.loads(Path(args.attribution_evidence).read_text(encoding="utf-8"))
        if attribution.get("classification") != "nonlinear_latent_transport_candidate":
            raise ValueError("flow preflight requires nonlinear latent transport attribution")
        device = torch.device(args.device)
        encoder, decoder = _load_wae(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        posterior = _posterior(encoder, args.dataset_root, device)
        labels = posterior["train_labels"].cpu()
        flow = LatentTransportFlow().to(device)
        optimizer = torch.optim.AdamW(flow.parameters(), lr=1.0e-3)
        before_flow, before_encoder, before_decoder = _clone_state(flow), _clone_state(encoder), _clone_state(decoder)
        initial_flow = _clone_state(flow)
        initial_cpu_rng = torch.get_rng_state()
        initial_cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        schedule = balanced_epoch_schedule(labels, seed=args.seed)
        updates = [_flow_update(flow, optimizer, posterior["train_codes"].index_select(0, schedule[index]).to(device), labels.index_select(0, schedule[index]).to(device)) for index in range(3)]
        after_flow = _clone_state(flow)
        uninterrupted_flow = _clone_state(flow)
        uninterrupted_optimizer = optimizer.state_dict()
        interrupted = LatentTransportFlow().to(device)
        interrupted.load_state_dict(initial_flow, strict=True)
        interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=1.0e-3)
        torch.set_rng_state(initial_cpu_rng)
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(initial_cuda_rng)
        batch_codes = posterior["train_codes"].index_select(0, schedule[0]).to(device)
        batch_labels = labels.index_select(0, schedule[0]).to(device)
        _flow_update(interrupted, interrupted_optimizer, batch_codes, batch_labels)
        interrupted_rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        resumed = LatentTransportFlow().to(device)
        resumed.load_state_dict(_clone_state(interrupted), strict=True)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1.0e-3)
        resumed_optimizer.load_state_dict(interrupted_optimizer.state_dict())
        torch.set_rng_state(interrupted_rng["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(interrupted_rng["cuda"])
        for index in (1, 2):
            _flow_update(resumed, resumed_optimizer, posterior["train_codes"].index_select(0, schedule[index]).to(device), labels.index_select(0, schedule[index]).to(device))
        replay_exact = _state_equal(uninterrupted_flow, _clone_state(resumed)) and _optimizer_equal(uninterrupted_optimizer, resumed_optimizer.state_dict())
        flow_changed = not _state_equal(before_flow, after_flow)
        wae_unchanged = _state_equal(_clone_state(encoder), before_encoder) and _state_equal(_clone_state(decoder), before_decoder)
        payload = {"schema": f"{SCHEMA_PREFIX}_preflight.v1", "status": "valid" if all(math.isfinite(value) for value in updates) and flow_changed and wae_unchanged and replay_exact else "invalid_harness", "updates": updates, "flow_changed": flow_changed, "wae_unchanged": wae_unchanged, "interrupted_resumed_replay_exact": replay_exact, "evaluator_calls": 0, "attribution_sha256": file_sha256(args.attribution_evidence)}
        _write_checked(output, payload)
        return 0
    if args.research_action == "screen":
        preflight = json.loads(Path(args.preflight_evidence).read_text(encoding="utf-8"))
        if preflight.get("status") != "valid":
            raise ValueError("latent flow screen requires valid preflight")
        device = torch.device(args.device)
        encoder, decoder = _load_wae(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        posterior = _posterior(encoder, args.dataset_root, device)
        flow_path, training_path, details = _flow_train(train_codes=posterior["train_codes"], train_labels=posterior["train_labels"], device=device, seed=args.seed, epochs=1, evidence_path=output, resume=args.resume_checkpoint)
        flow = _load_flow(flow_path, device)
        validation_nll = float(_flow_nll(flow, posterior["validation_codes"].to(device), posterior["validation_labels"].to(device)).cpu())
        statistics = _stats(posterior["train_codes"], posterior["train_labels"])
        validation_labels = posterior["validation_labels"].to(device)
        epsilon = torch.randn(posterior["validation_codes"].shape, generator=torch.Generator(device="cpu").manual_seed(8401)).to(device)
        with torch.no_grad():
            generated = flow(epsilon, validation_labels).cpu()
        p0_distance = _sliced_distance(epsilon.cpu(), posterior["validation_codes"])
        aclass_distance = _sliced_distance(_transport(epsilon, validation_labels, statistics, "Aclass").cpu(), posterior["validation_codes"])
        flow_distance = _sliced_distance(generated, posterior["validation_codes"])
        standard_nll = float((0.5 * posterior["validation_codes"].square().sum(1) + 0.5 * CODE_DIM * math.log(2.0 * math.pi)).mean())
        fit_passed = validation_nll < standard_nll and flow_distance <= 0.70 * p0_distance and math.isfinite(validation_nll)
        payload = {"schema": f"{SCHEMA_PREFIX}_screen.v1", "status": "valid", "passed": fit_passed, "flow_checkpoint": str(flow_path), "training_checkpoint": str(training_path), "details": details, "validation_nll": validation_nll, "standard_nll": standard_nll, "sliced_distance": {"P0": p0_distance, "Aclass": aclass_distance, "flow": flow_distance}, "evaluator_calls": 0, "lineage_sha256": canonical_sha256(lineage)}
        _write_checked(output, payload)
        return 0
    if args.research_action == "confirm":
        screen = json.loads(Path(args.screen_evidence).read_text(encoding="utf-8"))
        if screen.get("status") != "valid" or not screen.get("passed"):
            raise ValueError("latent flow confirm requires valid screen")
        device = torch.device(args.device)
        encoder, decoder = _load_wae(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        posterior = _posterior(encoder, args.dataset_root, device)
        flow_path, training_path, details = _flow_train(train_codes=posterior["train_codes"], train_labels=posterior["train_labels"], device=device, seed=args.seed, epochs=10, evidence_path=output, resume=args.resume_checkpoint)
        measurement = _flow_measure(flow_path, decoder, device, args.dataset_root, args.primary_evaluator_checkpoint, args.secondary_evaluator_checkpoint)
        classification = "electronic_latent_transport_admitted" if measurement["passed"] else "latent_flow_semantic_failed"
        payload = {"schema": f"{SCHEMA_PREFIX}_confirm.v1", "status": "valid", "flow_checkpoint": str(flow_path), "training_checkpoint": str(training_path), "training": details, "measurement": measurement, "classification": classification, "next_action": "register_o0_g_with_frozen_latent_transport" if measurement["passed"] else "close_current_electronic_and_fiber_route", "lineage_sha256": canonical_sha256(lineage), "formal_promotion": False, "training_authorized": False, "synthetic_diagnostic_only": True}
        _write_checked(output, payload)
        return 0
    if args.research_action == "locked-final":
        evidence_paths = [args.confirm_evidence, *args.peer_confirm_evidence]
        confirmations = [json.loads(Path(path).read_text(encoding="utf-8")) for path in evidence_paths]
        if sum(value.get("classification") == "electronic_latent_transport_admitted" for value in confirmations) < 2:
            raise ValueError("locked-final requires at least two admitted flow confirmations")
        device = torch.device(args.device)
        _, decoder = _load_wae(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        measurement = _flow_measure(args.flow_checkpoint, decoder, device, args.dataset_root, args.primary_evaluator_checkpoint, args.secondary_evaluator_checkpoint)
        classification = "electronic_latent_transport_admitted" if measurement["passed"] else "latent_flow_semantic_failed"
        payload = {"schema": f"{SCHEMA_PREFIX}_locked-final.v1", "status": "valid", "measurement": measurement, "classification": classification, "next_action": "register_o0_g_with_frozen_latent_transport" if measurement["passed"] else "close_current_electronic_and_fiber_route", "lineage_sha256": canonical_sha256(lineage), "confirm_evidence_sha256": [file_sha256(path) for path in evidence_paths], "formal_promotion": False, "training_authorized": False, "synthetic_diagnostic_only": True}
        _write_checked(output, payload)
        return 0
    if args.research_action == "closeout":
        locked = json.loads(Path(args.locked_evidence).read_text(encoding="utf-8"))
        if locked.get("schema") != f"{SCHEMA_PREFIX}_locked-final.v1" or locked.get("status") != "valid":
            raise ValueError("latent transport closeout requires valid locked-final evidence")
        payload = {"schema": f"{SCHEMA_PREFIX}_closeout.v1", "status": locked.get("status"), "classification": locked.get("classification", "invalid_harness"), "next_action": locked.get("next_action", "repair_and_revalidate_on_a_new_clean_commit"), "locked_evidence_sha256": file_sha256(args.locked_evidence), "formal_promotion": False, "training_authorized": False, "synthetic_diagnostic_only": True}
        _write_checked(output, payload)
        return 0
    raise RuntimeError(f"unsupported latent transport action {args.research_action}")
