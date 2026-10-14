"""Bounded 400x400 MNIST semantic-generation engineering baseline.

This project-designed diagnostic is not an exact reproduction of Chen et al.
The input/OLS/generator roles match the LightGen proxy, while the loss and
class-conditional diagonal Gaussian sampler are declared choices here.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.shared.research_run_control import CORRECTNESS_INVARIANTS, HeartbeatWriter, IMPLEMENTATION_SELECTION, RESOURCE_ENVELOPE, PreflightReceipt, ResearchRunControl, ResearchRunIdentity, capture_execution_source_manifest
from ldm_project.v2.config import load_config
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, file_sha256
from ldm_project.v2.optics.lightgen import LightGenAmplitudeFiberOLSSystemV2
from ldm_project.v2.student.evaluator import load_frozen_evaluator, stratified_mnist_indices
from ldm_project.v2.application.research.lightgen.confirmation_evaluators import ED_CHECKPOINT_SCHEMA as ED_SCHEMA, PixelMLPConfirmationEvaluatorV1


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_400_MNIST_SEMANTIC_GENERATION_ENGINEERING_BASELINE_V1"
OUTER_GATE = "LIGHTGEN_400_MNIST_GENERATION_CANARY_RESOLVED"
AUTHORIZATION = "USER_400_MNIST_SEMANTIC_GENERATION_VS_CURRENT_M0"
RUN_DIR = ROOT / "outputs/20260928_lightgen_400_mnist_semantic_generation_baseline_v1/runs/attempt_01"
CONFIG = Path("configs/v2/diagnostic/lightgen_amplitude_fiber_ols_400.yaml")
DATA = Path("data/MNIST/processed/training.pt")
TRAIN_EVALUATOR = Path("outputs/20260724_v2/runs/20260724_162422_705459_formal_v2_identity_logit_amplitude_finite_crop_identity/frozen_mnist_evaluator.pt")
ED_EVALUATOR = Path("outputs/20260920_sic_independent_confirmation_evaluator_v1/runs/attempt_01/final_epoch_candidate.pt")
M0_REPORT = Path("outputs/20260925_m0_bogt_semantic_generation_canary_v1/runs/attempt_01/resolved_report_05.json")
UPDATES = 600
BATCH_SIZE = 10
FIT_PER_CLASS = 200
GENERATE_PER_CLASS = 200
LEARNING_RATE = 5.0e-5
MACRO_RECALL_MIN = 0.50
CLASS_RECALL_MIN = 0.10
SATURATED_PIXEL_FRACTION_MAX = 0.01


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.nn.Module, torch.nn.Module]:
    for path in (DATA, TRAIN_EVALUATOR, ED_EVALUATOR, M0_REPORT):
        _require((ROOT / path).is_file(), f"required input is unavailable: {path}")
    payload = torch.load(ROOT / DATA, map_location="cpu", weights_only=False)
    _require(isinstance(payload, (tuple, list)) and len(payload) == 2, "MNIST TRAIN source schema changed")
    pixels, labels = payload
    _require(isinstance(pixels, torch.Tensor) and isinstance(labels, torch.Tensor), "MNIST tensors are missing")
    _require(pixels.shape == (60_000, 28, 28) and pixels.dtype == torch.uint8, "MNIST images changed")
    _require(labels.shape == (60_000,) and labels.dtype == torch.long, "MNIST labels changed")
    _require(int(labels.min()) == 0 and int(labels.max()) == 9, "MNIST labels are outside ten classes")
    train, dev = stratified_mnist_indices(labels, seed=1234, validation_per_label=500)
    train, dev = train.sort().values, dev.sort().values
    _require(train.numel() == 55_000 and dev.numel() == 5_000, "MNIST split count changed")
    _require(not bool(torch.isin(train, dev).any()), "TRAIN and DEV overlap")
    _require(torch.equal(torch.bincount(labels[dev], minlength=10), torch.full((10,), 500)), "DEV class count changed")
    images = pixels.unsqueeze(1).to(torch.float32).div_(255.0)
    training_evaluator, _ = load_frozen_evaluator(ROOT / TRAIN_EVALUATOR, device=device)
    training_evaluator.eval()
    for parameter in training_evaluator.parameters():
        parameter.requires_grad_(False)
    ed_payload = torch.load(ROOT / ED_EVALUATOR, map_location="cpu", weights_only=False)
    _require(isinstance(ed_payload, dict) and ed_payload.get("schema") == ED_SCHEMA, "E_D checkpoint schema changed")
    _require(ed_payload.get("authority_status") == "QUALIFIED", "E_D is not qualified")
    _require(ed_payload.get("independence", {}).get("role") == "INDEPENDENT_CONFIRMATION_ONLY", "E_D role changed")
    ed = PixelMLPConfirmationEvaluatorV1().to(device)
    ed.load_state_dict(ed_payload["state_dict"])
    ed.freeze()
    return images, labels, train, dev, training_evaluator, ed


def _balanced_rows(train: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(1234)
    by_class = []
    for digit in range(10):
        candidates = train[labels[train] == digit]
        _require(candidates.numel() >= UPDATES + FIT_PER_CLASS, f"TRAIN class {digit} is too small")
        by_class.append(candidates[torch.randperm(candidates.numel(), generator=generator)[:UPDATES]])
    rows = torch.stack(by_class, dim=1)
    _require(rows.shape == (UPDATES, BATCH_SIZE), "balanced schedule changed")
    return rows


def _input_and_target(images: torch.Tensor, indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    selected = images.index_select(0, indices.cpu()).to(device)
    target = F.interpolate(selected, size=(400, 400), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    return target.sqrt(), target


def _view(raw: torch.Tensor) -> torch.Tensor:
    _require(raw.ndim == 4 and tuple(raw.shape[1:]) == (1, 400, 400), "CMOS geometry changed")
    _require(bool(torch.isfinite(raw).all()), "raw CMOS has nonfinite values")
    return F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))


def _loss(model: LightGenAmplitudeFiberOLSSystemV2, evaluator: torch.nn.Module, amplitude: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, semantic_scale: float) -> tuple[torch.Tensor, dict[str, float]]:
    result = model(model.command(amplitude))
    raw = result.raw_camera.tensor
    rec = F.mse_loss(raw, target)
    ce = F.cross_entropy(evaluator(_view(raw))[1], labels)
    total = rec + semantic_scale * ce
    _require(bool(torch.isfinite(total)), "training loss is nonfinite")
    return total, {"reconstruction_mse": float(rec.detach()), "semantic_ce": float(ce.detach())}


def _step(model: LightGenAmplitudeFiberOLSSystemV2, optimizer: torch.optim.Optimizer, evaluator: torch.nn.Module, amplitude: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, semantic_scale: float) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    total, metrics = _loss(model, evaluator, amplitude, target, labels, semantic_scale)
    total.backward()
    for name, parameter in model.named_parameters():
        _require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), f"missing/nonfinite gradient: {name}")
        _require(bool(parameter.grad.abs().sum() > 0), f"zero gradient: {name}")
    _require(all(parameter.grad is None for parameter in evaluator.parameters()), "frozen training evaluator received gradients")
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.step()
    for name, parameter in model.named_parameters():
        _require(bool(torch.isfinite(parameter).all()), f"nonfinite parameter: {name}")
        _require(not torch.equal(parameter, before[name]), f"trainable group did not update: {name}")
    return {**metrics, "total": float(total.detach())}


def _optimizer(model: LightGenAmplitudeFiberOLSSystemV2) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def _memory_fraction(device: torch.device) -> float:
    free, total = torch.cuda.mem_get_info(device)
    _require(total > 0 and 0 <= free <= total, "CUDA reported invalid device memory")
    return max((total - free) / total, torch.cuda.max_memory_reserved(device) / total)


def _preflight(model: LightGenAmplitudeFiberOLSSystemV2, images: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor, evaluator: torch.nn.Module, device: torch.device) -> dict[str, float | bool]:
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    initial_gradients = {name: None if parameter.grad is None else parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    evaluator_state = {name: value.detach().clone() for name, value in evaluator.state_dict().items()}
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    updates = []
    try:
        label_batch = torch.arange(10, device=device)
        _require(torch.equal(labels.index_select(0, rows[0]).to(device), label_batch), "preflight rows do not match their declared labels")
        amplitude, target = _input_and_target(images, rows[0], device)
        with torch.no_grad():
            clean_logits = evaluator(images.index_select(0, rows[0]).to(device))[1]
            true_ce = F.cross_entropy(clean_logits, label_batch)
            shifted_ce = F.cross_entropy(clean_logits, (label_batch + 1) % 10)
            negative_control = bool(torch.isfinite(true_ce) and torch.isfinite(shifted_ce) and true_ce < shifted_ce)
            _require(negative_control, "shifted-label negative control did not reject wrong labels")
            torch.cuda.synchronize(device)
            real_started = time.perf_counter()
            real_result = model(model.command(amplitude))
            torch.cuda.synchronize(device)
            real_forward_seconds = time.perf_counter() - real_started
            generated_started = time.perf_counter()
            generated_result = model.forward_from_ols(real_result.ols.coefficients)
            torch.cuda.synchronize(device)
            generated_forward_seconds = time.perf_counter() - generated_started
            _require(torch.allclose(real_result.raw_camera.tensor, generated_result.raw_camera.tensor), "OLS direct path differs from real path")
            raw = real_result.raw_camera.tensor
            rec0 = F.mse_loss(raw, target)
            ce0 = F.cross_entropy(evaluator(_view(raw))[1], label_batch)
            semantic_scale = float((rec0 / ce0.clamp_min(1.0e-12)).detach())
            _require(math.isfinite(semantic_scale) and semantic_scale > 0.0, "semantic loss scale is invalid")
        optimizer = _optimizer(model)
        for cursor in range(3):
            _require(torch.equal(labels.index_select(0, rows[cursor]).to(device), label_batch), "preflight rows do not match their declared labels")
            amplitude, target = _input_and_target(images, rows[cursor], device)
            updates.append(_step(model, optimizer, evaluator, amplitude, target, label_batch, semantic_scale))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        fraction = _memory_fraction(device)
        estimated = (
            elapsed / 3.0 * UPDATES
            + real_forward_seconds * ((10 * FIT_PER_CLASS + 5_000) / BATCH_SIZE)
            + generated_forward_seconds * (10 * GENERATE_PER_CLASS / BATCH_SIZE)
            + 300.0
        )
        frozen_evaluator = bool(
            all(torch.equal(value, evaluator_state[name]) for name, value in evaluator.state_dict().items())
            and all(not parameter.requires_grad and parameter.grad is None for parameter in evaluator.parameters())
        )
        _require(frozen_evaluator, "training evaluator changed or is not frozen")
    finally:
        model.load_state_dict(initial_state)
        evaluator.load_state_dict(evaluator_state)
        for name, parameter in model.named_parameters():
            parameter.grad = initial_gradients[name]
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    model_restored = all(torch.equal(value, initial_state[name]) for name, value in model.state_dict().items())
    rng_restored = torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(before, after) for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else [], strict=True))
    _require(model_restored and rng_restored, "preflight failed to restore model/RNG")
    return {
        "seconds": elapsed, "real_forward_seconds": real_forward_seconds,
        "generated_forward_seconds": generated_forward_seconds, "peak_vram_fraction": fraction,
        "estimated_execution_seconds": estimated, "semantic_scale": semantic_scale,
        "negative_control_true_ce": float(true_ce), "negative_control_shifted_ce": float(shifted_ce),
        "three_updates": len(updates) == 3, "negative_control": negative_control,
        "frozen_evaluator": frozen_evaluator, "model_restored": model_restored, "rng_restored": rng_restored,
    }


def _class_ols_statistics(model: LightGenAmplitudeFiberOLSSystemV2, images: torch.Tensor, labels: torch.Tensor, train: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means = []
    real_std = []
    imag_std = []
    with torch.no_grad():
        for digit in range(10):
            candidates = train[labels[train] == digit][:FIT_PER_CLASS]
            values = []
            for chunk in candidates.split(BATCH_SIZE):
                amplitude, _ = _input_and_target(images, chunk, device)
                values.append(model(model.command(amplitude)).ols.coefficients.detach().cpu())
            coefficients = torch.cat(values)
            means.append(coefficients.mean(0))
            real_std.append(coefficients.real.std(0, unbiased=False).clamp_min(1.0e-8))
            imag_std.append(coefficients.imag.std(0, unbiased=False).clamp_min(1.0e-8))
    return torch.stack(means), torch.stack(real_std), torch.stack(imag_std)


def _confusion(predictions: torch.Tensor, labels: torch.Tensor) -> tuple[list[list[int]], list[float], float]:
    confusion = torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
    row_counts = confusion.sum(1)
    _require(bool((row_counts > 0).all()), "evaluation misses a digit class")
    recall = confusion.diag().to(torch.float64) / row_counts
    return confusion.tolist(), recall.tolist(), float(recall.mean())


def _m0_reference() -> dict[str, object]:
    payload = json.loads((ROOT / M0_REPORT).read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m0_bogt_semantic_generation_report.v1", "M0 report schema changed")
    _require(payload.get("outer_gate") == "M0_BOGT_SEMANTIC_GENERATION_CANARY_RESOLVED" and payload.get("classification") == "VALID_FAILED", "M0 report authority changed")
    generated = payload["generated_outputs"]["independent_E_D"]
    _require(generated.get("evaluator") == "E_D independent confirmation evaluator v1", "M0 evaluator changed")
    _require(generated.get("requested_count") == 2_000 and generated.get("per_class_count") == 200, "M0 requested bank changed")
    return {"report": M0_REPORT.as_posix(), "sha256": file_sha256(ROOT / M0_REPORT), "generated_s1_ed_macro_recall": generated["macro_recall"], "recall_by_class": generated["per_class_recall"], "comparison_role": "descriptive_only_different_geometry_and_sampler"}


def _evaluate_dev(model: LightGenAmplitudeFiberOLSSystemV2, ed: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, dev: torch.Tensor, device: torch.device) -> dict[str, object]:
    predictions = []
    with torch.no_grad():
        for chunk in dev.split(BATCH_SIZE):
            amplitude, _ = _input_and_target(images, chunk, device)
            raw = model(model.command(amplitude)).raw_camera.tensor
            predictions.append(ed(_view(raw)).argmax(1).cpu())
    confusion, recall, macro = _confusion(torch.cat(predictions), labels[dev])
    return {"count": int(dev.numel()), "confusion": confusion, "recall_by_class": recall, "macro_recall": macro}


def _evaluate_generation(model: LightGenAmplitudeFiberOLSSystemV2, ed: torch.nn.Module, means: torch.Tensor, real_std: torch.Tensor, imag_std: torch.Tensor, device: torch.device) -> tuple[dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260928)
    predictions = []
    requested = []
    sampled = []
    evaluator_views = []
    raw_examples = []
    saturation_pixels = 0
    total_pixels = 0
    raw_min = math.inf
    raw_max = -math.inf
    with torch.no_grad():
        for digit in range(10):
            for batch_index in range(GENERATE_PER_CLASS // BATCH_SIZE):
                re = torch.randn((BATCH_SIZE, 100), generator=generator) * real_std[digit]
                im = torch.randn((BATCH_SIZE, 100), generator=generator) * imag_std[digit]
                z = (means[digit] + torch.complex(re, im)).to(device)
                raw = model.forward_from_ols(z).raw_camera.tensor
                view = _view(raw)
                predictions.append(ed(view).argmax(1).cpu())
                requested.append(torch.full((BATCH_SIZE,), digit, dtype=torch.long))
                sampled.append(z.detach().cpu())
                evaluator_views.append(view.detach().cpu())
                if batch_index == 0:
                    raw_examples.append(raw[:5].detach().cpu())
                saturation_pixels += int((raw > 1.0).sum())
                total_pixels += raw.numel()
                raw_min = min(raw_min, float(raw.min()))
                raw_max = max(raw_max, float(raw.max()))
    confusion, recall, macro = _confusion(torch.cat(predictions), torch.cat(requested))
    metrics = {"count": 10 * GENERATE_PER_CLASS, "confusion": confusion, "recall_by_class": recall, "macro_recall": macro, "saturated_pixel_fraction": saturation_pixels / total_pixels, "raw_min": raw_min, "raw_max": raw_max}
    return metrics, torch.cat(sampled), torch.cat(evaluator_views), torch.cat(raw_examples)


def main() -> None:
    _require(torch.cuda.is_available(), "400x400 baseline requires CUDA")
    _require(not RUN_DIR.exists(), f"run path already exists: {RUN_DIR}")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    device = torch.device("cuda")
    config = load_config(ROOT / CONFIG)
    _require(config.system.kind == "diagnostic_lightgen_amplitude_fiber_ols_400", "400x400 config changed")
    images, labels, train, dev, evaluator, ed = _load_inputs(device)
    rows = _balanced_rows(train, labels)
    model = LightGenAmplitudeFiberOLSSystemV2(config.system).to(device)
    model.train()
    manifest = capture_execution_source_manifest(
        ROOT,
        task_id=TASK_ID,
        semantic_authority_id=OUTER_GATE,
        relevant_paths=(
            Path(__file__), CONFIG, "src/ldm_project/v2/optics/lightgen.py",
            "src/ldm_project/v2/optics/fiber_ols.py", "src/ldm_project/v2/optics/d2nn.py",
            "src/ldm_project/v2/student/evaluator.py",
        ),
        config_identity={"geometry": "400x400/2+4/100-complex-OLS", "updates": UPDATES, "batch": BATCH_SIZE, "inputs": [DATA.as_posix(), TRAIN_EVALUATOR.as_posix(), ED_EVALUATOR.as_posix(), M0_REPORT.as_posix()]},
    )
    identity = ResearchRunIdentity(
        task_id=TASK_ID, run_id="lightgen_400_mnist_semantic_baseline_v1", trajectory_id="fixed_theta0_seed1234",
        semantic_authority_id=OUTER_GATE, data_role="TRAIN_55000_DEV_5000_NO_TEST",
        config_identity={"config": CONFIG.as_posix(), "updates": UPDATES},
        model_identity="diagnostic_lightgen_amplitude_fiber_ols_400",
        optimizer_identity={"kind": "Adam", "lr": LEARNING_RATE},
        sampler_identity={"kind": "class_diagonal_complex_gaussian", "fit_per_class": FIT_PER_CLASS, "seed": 20260928},
        execution_source_semantics_id=manifest.execution_source_semantics_id,
        source_manifest_id=manifest.source_manifest_id,
    )
    control = ResearchRunControl(identity, authorization_id=AUTHORIZATION)
    preflight = _preflight(model, images, labels, rows, evaluator, device)
    receipt = PreflightReceipt(
        source_semantics_id=manifest.execution_source_semantics_id,
        authorization_id=AUTHORIZATION,
        checks={
            CORRECTNESS_INVARIANTS: {
                "passed": all(preflight[key] is True for key in ("three_updates", "negative_control", "frozen_evaluator", "model_restored", "rng_restored")),
                **{key: preflight[key] for key in ("three_updates", "negative_control", "frozen_evaluator", "model_restored", "rng_restored")},
            },
            RESOURCE_ENVELOPE: {
                "passed": preflight["three_updates"] is True and math.isfinite(preflight["peak_vram_fraction"]) and 0 <= preflight["peak_vram_fraction"] <= 1,
                "peak_vram_fraction": preflight["peak_vram_fraction"],
                "estimated_execution_seconds": preflight["estimated_execution_seconds"],
            },
            IMPLEMENTATION_SELECTION: {
                "passed": device.type == "cuda" and config.system.kind == "diagnostic_lightgen_amplitude_fiber_ols_400" and preflight["three_updates"] is True,
                "engine": "full_400x400_cuda",
            },
        },
        selected_engine="full_400x400_cuda", resource={"peak_vram_fraction_observed": preflight["peak_vram_fraction"]},
    )
    control.mark_preflight_ready(receipt)
    control.open_scientific_attempt(RUN_DIR, attempt_id="attempt_01", receipt=receipt)
    atomic_write_new_json(RUN_DIR / "preflight.json", {"receipt": receipt.to_dict(), "measurements": preflight, "source_manifest": manifest.to_dict()})
    heartbeat = HeartbeatWriter(RUN_DIR, identity.run_id, total_steps=UPDATES)
    heartbeat.update(state=control.state, phase="TRAINING", step=0, force=True)
    heartbeat.start_liveness()
    optimizer = _optimizer(model)
    evaluator_state = {name: value.detach().clone() for name, value in evaluator.state_dict().items()}
    ed_state = {name: value.detach().clone() for name, value in ed.state_dict().items()}
    started = time.perf_counter()
    history = []
    label_batch = torch.arange(10, device=device)
    for cursor, row in enumerate(rows, start=1):
        amplitude, target = _input_and_target(images, row, device)
        history.append(_step(model, optimizer, evaluator, amplitude, target, label_batch, float(preflight["semantic_scale"])))
        heartbeat.update(state=control.state, phase="TRAINING", step=cursor, latest_losses=history[-1])
        if cursor % 25 == 0:
            heartbeat.raise_liveness_error()
            fraction = _memory_fraction(device)
            atomic_write_new_json(RUN_DIR / f"progress_{cursor:04d}.json", {"completed_updates": cursor, "latest_loss": history[-1], "elapsed_seconds": time.perf_counter() - started, "peak_vram_fraction": fraction})
            print(f"step {cursor}/{UPDATES}: loss={history[-1]['total']:.6g}, peak VRAM={fraction:.3f}", flush=True)
    model.eval()
    _require(all(torch.equal(value, evaluator_state[name]) for name, value in evaluator.state_dict().items()), "training evaluator changed")
    _require(all(torch.equal(value, ed_state[name]) for name, value in ed.state_dict().items()), "independent E_D changed")
    heartbeat.update(state=control.state, phase="POST_TRAIN_FIT", step=UPDATES, force=True)
    means, real_std, imag_std = _class_ols_statistics(model, images, labels, train, device)
    dev_metrics = _evaluate_dev(model, ed, images, labels, dev, device)
    generated_metrics, sampled_ols, generated_views, raw_examples = _evaluate_generation(model, ed, means, real_std, imag_std, device)
    passed = generated_metrics["macro_recall"] >= MACRO_RECALL_MIN and min(generated_metrics["recall_by_class"]) >= CLASS_RECALL_MIN and generated_metrics["saturated_pixel_fraction"] <= SATURATED_PIXEL_FRACTION_MAX
    if passed:
        control.complete()
    else:
        control.valid_failed()
    heartbeat.update(state=control.state, phase="FINALIZING", step=UPDATES, force=True)
    atomic_torch_save_new(RUN_DIR / "model_step_000600.pt", {"schema": "lightgen_400_mnist_semantic_baseline_checkpoint.v1", "model": {name: value.detach().cpu() for name, value in model.state_dict().items()}, "completed_updates": UPDATES, "sampler": {"class_mean": means, "class_real_std": real_std, "class_imag_std": imag_std, "fit_per_class": FIT_PER_CLASS, "sample_seed": 20260928}})
    atomic_torch_save_new(RUN_DIR / "generated_ols.pt", {"schema": "lightgen_400_mnist_generated_ols.v1", "coefficients": sampled_ols, "requested_labels": torch.arange(10).repeat_interleave(GENERATE_PER_CLASS)})
    atomic_torch_save_new(RUN_DIR / "generated_visuals.pt", {"schema": "lightgen_400_mnist_generated_visuals.v1", "evaluator_views_28x28": generated_views, "raw_cmos_examples_400x400": raw_examples, "requested_labels": torch.arange(10).repeat_interleave(GENERATE_PER_CLASS), "examples_per_class": 5})
    report = {"schema": "lightgen_400_mnist_semantic_baseline_report.v1", "evidence_tier": "diagnostic-only", "paper_reproduction": False, "outer_gate": OUTER_GATE, "status": control.state, "preflight": preflight, "source_manifest": manifest.to_dict(), "train": {"updates": UPDATES, "initial_loss": history[0], "final_loss": history[-1], "semantic_scale": preflight["semantic_scale"]}, "dev_real_path": dev_metrics, "generated": generated_metrics, "gate": {"macro_recall_min": MACRO_RECALL_MIN, "class_recall_min": CLASS_RECALL_MIN, "saturated_pixel_fraction_max": SATURATED_PIXEL_FRACTION_MAX, "passed": passed}, "m0_reference": _m0_reference(), "artifacts": {"model": "model_step_000600.pt", "generated_ols": "generated_ols.pt", "generated_visuals": "generated_visuals.pt"}, "elapsed_seconds": time.perf_counter() - started, "peak_vram_fraction": _memory_fraction(device), "protected_test_reads": 0}
    heartbeat.raise_liveness_error()
    atomic_write_new_json(RUN_DIR / "report.json", report)
    heartbeat.update(state=control.state, phase=control.state, step=UPDATES, force=True)
    heartbeat.stop_liveness()
    print(json.dumps({"status": control.state, "generated_macro_recall": generated_metrics["macro_recall"], "dev_real_macro_recall": dev_metrics["macro_recall"], "report": str(RUN_DIR / "report.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
