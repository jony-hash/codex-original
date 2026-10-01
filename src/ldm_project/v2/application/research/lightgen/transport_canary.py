"""Bounded CUDA diagnostics for the electronic transport pilots."""
from __future__ import annotations

import shutil
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from ldm_project.v2.application.research.capacity.electronic_o0_latent_transport import CODE_DIM, LatentTransportFlow, _clone_state, _flow_update, _optimizer_equal, _state_equal
from ldm_project.v2.application.research.capacity.electronic_o0_wae import balanced_epoch_schedule, panel_codes
from ldm_project.v2.application.research.shared.lifecycle_artifacts import load_posterior, write_posterior


def _balanced_rows(labels: torch.Tensor, count: int = 500) -> torch.Tensor:
    rows = []
    per_class = count // 10
    for label in range(10):
        selected = torch.nonzero(labels.cpu() == label, as_tuple=False).flatten()[:per_class]
        rows.append(selected)
    indices = torch.cat(rows)
    if indices.numel() != count:
        raise ValueError("canary validation set does not contain 500 balanced rows")
    return indices


def run_flow_canary(
    posterior: dict[str, torch.Tensor],
    *,
    device: torch.device,
    temporary_root: str | Path,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("electronic transport canary requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if tuple(posterior["train_codes"].shape) != (55000, CODE_DIM) or tuple(posterior["validation_codes"].shape) != (5000, CODE_DIM):
        raise ValueError("canary requires formal posterior dimensions")
    validation_rows = _balanced_rows(posterior["validation_labels"])
    torch.cuda.reset_peak_memory_stats(device)
    design_codes = posterior["validation_codes"].index_select(0, validation_rows).to(device)
    design_labels = posterior["validation_labels"].index_select(0, validation_rows).to(device)
    validation_codes = posterior["validation_codes"].to(device)
    validation_labels = posterior["validation_labels"].to(device)
    if design_codes.shape != (500, CODE_DIM) or design_labels.shape != (500,):
        raise ValueError("canary CPU-index/CUDA-gather contract failed")
    torch.cuda.empty_cache()
    torch.manual_seed(8401)
    torch.cuda.manual_seed_all(8401)
    torch.use_deterministic_algorithms(True)
    flow = LatentTransportFlow().to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=1.0e-3)
    initial_flow = _clone_state(flow)
    initial_rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
    schedule = balanced_epoch_schedule(design_labels.cpu(), seed=8401, per_class=5)
    updates = []
    for indices in schedule[:3]:
        updates.append(_flow_update(flow, optimizer, design_codes.index_select(0, indices.to(device)), design_labels.index_select(0, indices.to(device))))
    uninterrupted = _clone_state(flow)
    uninterrupted_optimizer = optimizer.state_dict()
    interrupted = LatentTransportFlow().to(device)
    interrupted.load_state_dict(initial_flow, strict=True)
    interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(), lr=1.0e-3)
    torch.set_rng_state(initial_rng[0])
    torch.cuda.set_rng_state_all(initial_rng[1])
    _flow_update(interrupted, interrupted_optimizer, design_codes.index_select(0, schedule[0].to(device)), design_labels.index_select(0, schedule[0].to(device)))
    replay_rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
    resumed = LatentTransportFlow().to(device)
    resumed.load_state_dict(_clone_state(interrupted), strict=True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1.0e-3)
    resumed_optimizer.load_state_dict(interrupted_optimizer.state_dict())
    torch.set_rng_state(replay_rng[0])
    torch.cuda.set_rng_state_all(replay_rng[1])
    for indices in schedule[1:3]:
        _flow_update(resumed, resumed_optimizer, design_codes.index_select(0, indices.to(device)), design_labels.index_select(0, indices.to(device)))
    replay_exact = _state_equal(uninterrupted, _clone_state(resumed)) and _optimizer_equal(uninterrupted_optimizer, resumed_optimizer.state_dict())
    with torch.no_grad():
        inference = flow(design_codes, design_labels)
    if inference.requires_grad:
        raise RuntimeError("canary inference tensor crossed the autograd boundary")
    if not torch.isfinite(inference).all() or not all(torch.isfinite(torch.tensor(value)) for value in updates):
        raise FloatingPointError("canary produced non-finite values")
    panel, panel_labels = panel_codes(4242, device=device)
    if tuple(panel.shape) != (500, CODE_DIM) or tuple(panel_labels.shape) != (500,):
        raise ValueError("canary panel shape contract failed")
    masked = validation_codes[validation_labels == validation_labels[0]]
    broadcast = design_codes + torch.zeros((1, CODE_DIM), device=device)
    mask = torch.ones_like(broadcast, dtype=torch.bool)
    if broadcast.shape != design_codes.shape or int(mask.sum()) != broadcast.numel():
        raise ValueError("canary broadcast/mask shape contract failed")
    quantile = torch.quantile(masked, 0.5, dim=0)
    if tuple(quantile.shape) != (CODE_DIM,):
        raise ValueError("canary quantile shape contract failed")
    torch.cuda.empty_cache()
    temp_root = Path(temporary_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="posterior-canary-", dir=temp_root) as directory:
        artifact = Path(directory) / "encoded_posterior.v1.pt"
        small = {name: value[:2].cpu() for name, value in posterior.items()}
        write_posterior(artifact, small, {"producer_contract": "canary", "domain": "electronic_latent"})
        load_posterior(artifact, expected={"producer_contract": "canary", "domain": "electronic_latent"})
        tampered = Path(directory) / "tampered.pt"
        shutil.copyfile(artifact, tampered)
        shutil.copyfile(artifact.with_suffix(".json"), tampered.with_suffix(".json"))
        tampered.write_bytes(tampered.read_bytes() + b"tamper")
        try:
            load_posterior(tampered, expected={"producer_contract": "canary"})
        except ValueError:
            tamper_rejected = True
        else:
            tamper_rejected = False
    torch.cuda.empty_cache()
    storage_available = shutil.disk_usage(temp_root).free > 64 * 1024 * 1024
    peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    storage = shutil.disk_usage(temp_root)
    storage_expected_bytes = 64 * 1024 * 1024
    storage_gate = storage.free >= storage_expected_bytes
    return {
        "schema": "electronic_o0_transport_canary.v1",
        "status": "valid" if replay_exact and tamper_rejected and storage_available and storage_gate else "invalid_harness",
        "diagnostic_only": True,
        "domain": "electronic_latent",
        "design_rows": int(design_codes.shape[0]),
        "validation_rows": int(validation_codes.shape[0]),
        "updates": updates,
        "replay_exact": replay_exact,
        "tamper_rejected": tamper_rejected,
        "panel_shape": list(panel.shape),
        "quantile_shape": list(quantile.shape),
        "storage_available": storage_available,
        "storage_gate": storage_gate,
        "peak_memory_bytes": peak_memory_bytes,
        "broadcast_shape": list(broadcast.shape),
        "mask_shape": list(mask.shape),
        "cache_reset": True,
    }
