"""Compare the frozen LightGen checkpoints by CMOS readability and trace fields.

This is a descriptive, read-only visualization adapter. It has two phases:
the first replays a frozen allowlist of 27 checkpoints and writes blinded CMOS
rating sheets; the second consumes the completed ratings CSV and renders the
best, worst, and median eight-node field traces. It never trains or updates a
model parameter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import secrets
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ldm_project.v2.config import load_config
from ldm_project.v2.evidence import file_sha256, safe_torch_load
from ldm_project.v2.optics.lightgen import LightGenAmplitudeFiberOLSSystemV2
from ldm_project.v2.application.research.lightgen import canonical_semantics
from scripts.research.lightgen import execute_lightgen_2d_evaluator_agnostic_structural_image_alignment_v1 as structural
from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as robust
from scripts.research.lightgen import execute_lightgen_2d_400_mnist_semantic_generation_baseline_v1 as cloud_baseline
from scripts.research.lightgen import execute_lightgen_2d_joint_structural_semantic_continuation_v1 as joint
from scripts.research.lightgen import execute_lightgen_2d_three_evaluator_robust_semantic_continuation_v1 as sic


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_RELATIVE = Path("outputs/20260923_cmos_eight_node_step_comparison_v1/runs/attempt_01")
UNBLINDING_KEY_NAME = "unblinding_key_open_only_after_ratings.json"
SELECTION_MANIFEST = Path(
    "outputs/20260921_step6000_conditional_generation_readiness_v1/runs/attempt_06/fixed_selection_manifest.json"
)
SELECTION_SOURCE_INDICES = "real_dev.canonical_source_indices"
SAMPLE_COUNT = 50
DISPLAY_PER_CLASS = 2
RATING_SEED = 20260923
FORWARD_BATCH_SIZE = 2
PI2 = 2.0 * math.pi

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
    "Encoder layer 2 / pre-OLS\n100×100",
    "OLS re-emission / W_OLS\n100×100 + z inset",
    "Generator layer 1\n100×100",
    "Generator layer 2\n100×100",
    "CMOS incident field\n200×200",
)

RUNS = (
    {
        "name": "semantic",
        "run": Path("outputs/20260920_evaluator_robust_semantic_training_v1/runs/attempt_01"),
        "report_trajectory": "long_horizon_trajectory",
        "checkpoint_schema": "evaluator_robust_semantic_training_checkpoint.v1",
        "steps": (600, 1000, 1500, 2000, 2500, 3000),
        "classification": "EVALUATOR_ROBUST_SEMANTIC_RECOVERY_SCALES_TO_LONG_HORIZON",
    },
    {
        "name": "sic",
        "run": Path("outputs/20260920_three_evaluator_robust_semantic_continuation_v1/runs/attempt_02"),
        "report_trajectory": "training_trajectory",
        "checkpoint_schema": "three_evaluator_robust_semantic_continuation_checkpoint.v1",
        "steps": (3100, 3200, 3300, 3600, 4000, 4500, 5000, 5500, 6000),
        "classification": "THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_SUPPORTED",
    },
    {
        "name": "structural",
        "run": Path("outputs/20260921_evaluator_agnostic_structural_image_alignment_v1/runs/attempt_03"),
        "report_trajectory": "training_trajectory",
        "checkpoint_schema": "evaluator_agnostic_structural_image_alignment_checkpoint.v1",
        "steps": (6100, 6200, 6300, 6600, 7000, 7500, 8000, 8500, 9000),
        "classification": "STRUCTURAL_RECOVERY_WITH_CLASSIFIER_SEMANTIC_TRADEOFF",
    },
    {
        "name": "canary",
        "run": Path("outputs/20260922_local_compute_bounded_joint_canary_v1/runs/attempt_02"),
        "report_trajectory": "training_trajectory",
        "checkpoint_schema": "lightgen_2d_local_compute_bounded_joint_canary_checkpoint.v1",
        "steps": (9100, 9200, 9300),
        "classification": "JOINT_SEMANTIC_RECOVERY_WITHOUT_CANONICAL_STRUCTURAL_TRANSFER",
    },
)
CANDIDATE_STEPS = tuple(step for run in RUNS for step in run["steps"])
ENDPOINT_TRACE_OUTPUT_RELATIVE = Path(
    "outputs/20260930_cloud_vs_m0_diffractive_comparison_v1/runs/phase_amplitude_02"
)
ENDPOINT_TRACE_SELECTION = SELECTION_MANIFEST
CLOUD_CONFIG = Path("configs/v2/diagnostic/lightgen_amplitude_fiber_ols_400.yaml")
CLOUD_CHECKPOINT = Path(
    "outputs/20260928_lightgen_400_mnist_semantic_generation_baseline_v1/"
    "runs/autodl_attempt_01/model_step_000600.pt"
)
CLOUD_GENERATED_OLS = Path(
    "outputs/20260928_lightgen_400_mnist_semantic_generation_baseline_v1/"
    "runs/autodl_attempt_01/generated_ols.pt"
)
CLOUD_GENERATED_VISUALS = Path(
    "outputs/20260928_lightgen_400_mnist_semantic_generation_baseline_v1/"
    "runs/autodl_attempt_01/generated_visuals.pt"
)
M0_TRACE_ENDPOINTS = (
    {
        "key": "m0_pure_bogt_step1200",
        "label": "M0 纯 BOGT step1200",
        "checkpoint": Path(
            "outputs/20260928_m0_bogt_joint_generation_canary_v1/"
            "runs/attempt_02/scientific_step_001200.pt"
        ),
        "status_path": Path("outputs/20260928_m0_bogt_joint_generation_canary_v1/runs/attempt_02/status.json"),
        "parent_state": "INVALID_HARNESS",
        "status": "INVALID_HARNESS parent attempt; descriptive endpoint only",
        "schema": "m0_bogt_joint_generation_canary_checkpoint.v1",
    },
    {
        "key": "m0_canonical_step1200",
        "label": "M0 + canonical step1200",
        "checkpoint": Path(
            "outputs/20260929_m0_bogt_canonical_gradient_augmentation_canary_v1/"
            "runs/attempt_02/scientific_step_001200.pt"
        ),
        "status_path": Path("outputs/20260929_m0_bogt_canonical_gradient_augmentation_canary_v1/runs/attempt_02/status.json"),
        "parent_state": "COMPLETED",
        "status": "VALID_COMPLETED descriptive endpoint",
        "schema": "m0_bogt_canonical_gradient_augmentation_checkpoint.v1",
    },
)
CLOUD_NODE_KEYS = (
    "slm_output",
    "encoder_layer_1",
    "encoder_layer_2",
    "fiber_input",
    "fiber_reemission",
    "decoder_layer_1",
    "decoder_layer_2",
    "decoder_layer_3",
    "decoder_layer_4",
    "cmos_field",
)
CLOUD_NODE_LABELS = (
    "SLM 输出场\n400×400",
    "编码层 1 后\n400×400",
    "编码层 2 后\n400×400",
    "光纤 / OLS 输入面\n400×400",
    "OLS 再发射场\n400×400 + 10×10 系数",
    "解码层 1 后\n400×400",
    "解码层 2 后\n400×400",
    "解码层 3 后\n400×400",
    "解码层 4 后\n400×400",
    "CMOS 入射场\n400×400",
)
M0_NODE_LABELS_ZH = (
    "SLM 输出场\n200×200",
    "前置 4f / 编码器输入\n100×100",
    "编码层 1 后\n100×100",
    "编码层 2 后 / OLS 输入\n100×100",
    "OLS 再发射场\n100×100 + 10×10 系数",
    "解码层 1 后\n100×100",
    "解码层 2 后\n100×100",
    "CMOS 入射场\n200×200",
)
EXPECTED_CANDIDATE_STEPS = (
    600, 1000, 1500, 2000, 2500, 3000,
    3100, 3200, 3300, 3600, 4000, 4500, 5000, 5500, 6000,
    6100, 6200, 6300, 6600, 7000, 7500, 8000, 8500, 9000,
    9100, 9200, 9300,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _candidate_registry(root: Path) -> list[dict[str, Any]]:
    _require(CANDIDATE_STEPS == EXPECTED_CANDIDATE_STEPS, "frozen candidate allowlist changed")
    candidates: list[dict[str, Any]] = []
    for run_spec in RUNS:
        run_dir = root / run_spec["run"]
        report_path = run_dir / "report.json"
        _require(report_path.is_file(), f"candidate run report is missing: {report_path}")
        report = _read_json(report_path)
        _require(report.get("result") == "VALID_COMPLETED", f"candidate run is not valid completed: {run_spec['name']}")
        _require(report.get("classification") == run_spec["classification"], f"candidate run classification changed: {run_spec['name']}")
        trajectory = report.get(run_spec["report_trajectory"])
        _require(isinstance(trajectory, dict), f"candidate trajectory missing: {run_spec['name']}")
        saved = trajectory.get("checkpoint_steps_persisted")
        _require(isinstance(saved, list), f"checkpoint schedule missing: {run_spec['name']}")
        _require(set(run_spec["steps"]).issubset(set(int(step) for step in saved)), f"frozen steps absent from producer report: {run_spec['name']}")
        for step in run_spec["steps"]:
            checkpoint_path = run_dir / f"scientific_step_{step:06d}.pt"
            _require(checkpoint_path.is_file(), f"candidate scientific checkpoint is missing: {checkpoint_path}")
            payload = safe_torch_load(checkpoint_path, map_location="cpu")
            _require(isinstance(payload, Mapping), f"candidate checkpoint is malformed: step{step}")
            _require(payload.get("schema") == run_spec["checkpoint_schema"], f"checkpoint schema changed at step{step}")
            _require(payload.get("completed_updates") == step, f"checkpoint cursor differs from filename at step{step}")
            _require(payload.get("scientific_artifact") is True and payload.get("operational_artifact") is False, f"step{step} is not a scientific checkpoint")
            _require(payload.get("trainable_parameter_names") == ["generator.phase"], f"trainable boundary changed at step{step}")
            states: dict[str, torch.Tensor] = {}
            for name in ("encoder_state", "generator_state"):
                tensor = payload.get(name)
                _require(torch.is_tensor(tensor) and tensor.shape == (2, 1, 100, 100) and tensor.dtype == torch.float32, f"invalid {name} at step{step}")
                _require(bool(torch.isfinite(tensor).all()), f"nonfinite {name} at step{step}")
                states[name] = tensor
            candidates.append(
                {
                    "step": step,
                    "run_name": run_spec["name"],
                    "run_dir": run_spec["run"].as_posix(),
                    "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "checkpoint_schema": run_spec["checkpoint_schema"],
                    "report_path": report_path.relative_to(root).as_posix(),
                    "report_sha256": file_sha256(report_path),
                    "run_classification": report["classification"],
                    "outer_gate_passed": report.get("outer_gate_passed"),
                    "states": states,
                }
            )
    _require(tuple(item["step"] for item in candidates) == CANDIDATE_STEPS, "candidate enumeration is not the frozen ordered allowlist")
    # The final canary completed execution but its scientific condition failed.
    canary_report = _read_json(root / RUNS[-1]["run"] / "report.json")
    _require(canary_report.get("outer_gate_passed") is False, "step9300 canary gate status changed")
    return candidates


def _load_selection(root: Path, authorities: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, list[int], dict[str, Any]]:
    path = root / SELECTION_MANIFEST
    _require(path.is_file(), f"accepted fixed DEV selection is missing: {path}")
    manifest = _read_json(path)
    selection = manifest.get("real_dev")
    _require(isinstance(selection, dict), "real DEV selection block is missing")
    source_indices = selection.get("canonical_source_indices")
    labels = selection.get("labels")
    positions = selection.get("positions_in_sorted_dev")
    _require(isinstance(source_indices, list) and len(source_indices) == SAMPLE_COUNT, "fixed DEV source index count changed")
    _require(isinstance(labels, list) and len(labels) == SAMPLE_COUNT, "fixed DEV label count changed")
    _require(isinstance(positions, list) and len(positions) == SAMPLE_COUNT, "fixed DEV position count changed")
    source = torch.tensor(source_indices, dtype=torch.long)
    expected = authorities["development_indices"].index_select(0, torch.tensor(positions, dtype=torch.long))
    _require(torch.equal(source, expected), "fixed visualization indices differ from accepted canonical DEV selection")
    observed_labels = authorities["labels"].index_select(0, source)
    _require(observed_labels.tolist() == labels, "fixed DEV labels differ from the accepted selection manifest")
    counts = torch.bincount(observed_labels, minlength=10)
    _require(counts.tolist() == [5] * 10, "fixed DEV selection is not exactly five examples per class")
    images = authorities["images"].index_select(0, source).contiguous()
    _require(images.shape == (SAMPLE_COUNT, 1, 28, 28), "fixed DEV input geometry changed")
    return images, observed_labels, source_indices, manifest


def _load_models(root: Path, authorities: Mapping[str, Any], device: torch.device) -> tuple[Any, dict[str, Any]]:
    consumer, smooth, independent, ec, ed = sic._load_models(root, authorities, device)
    models = {"E_S": smooth, "E_I": independent, "E_C": ec, "E_D": ed}
    for name, model in models.items():
        model.eval()
        _require(all(not parameter.requires_grad for parameter in model.parameters()), f"{name} is not frozen")
    return consumer, models


def _load_structural_normalization(root: Path, device: torch.device, train_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = root / joint.SURROGATE_NORMALIZATION
    _require(path.is_file(), f"TRAIN-only structural normalization is missing: {path}")
    payload = safe_torch_load(path, map_location="cpu")
    _require(isinstance(payload, Mapping) and payload.get("train_only") is True, "structural normalization role changed")
    mean = payload.get("mean")
    scale = payload.get("scale")
    _require(torch.is_tensor(mean) and mean.shape == (336,), "structural normalization mean geometry changed")
    _require(torch.is_tensor(scale) and scale.shape == (336,), "structural normalization scale geometry changed")
    recomputed_mean, recomputed_scale = structural._fit_surrogate_normalization(train_images)
    _require(torch.equal(recomputed_mean, mean) and torch.equal(recomputed_scale, scale), "accepted structural normalization differs from canonical TRAIN recomputation")
    verification = {
        "artifact_sha256": file_sha256(path),
        "canonical_train_source_sha256": robust.TRAINING_SOURCE_SHA256,
        "embedded_source_sha256_provenance_only": payload.get("source_sha256"),
        "recomputed_from_canonical_train_exact_match": True,
    }
    return mean.to(device), scale.to(device), verification


def _ranked_aliases(candidates: Sequence[Mapping[str, Any]]) -> tuple[dict[int, str], dict[str, int]]:
    rng = np.random.default_rng(RATING_SEED)
    order = list(range(len(candidates)))
    rng.shuffle(order)
    step_to_alias = {int(candidates[index]["step"]): f"K{rank + 1:02d}" for rank, index in enumerate(order)}
    alias_to_step = {alias: step for step, alias in step_to_alias.items()}
    return step_to_alias, alias_to_step


def _capture_batch(consumer: Any, batch: torch.Tensor) -> dict[str, torch.Tensor]:
    runtime = consumer.runtime
    relays: dict[str, list[torch.Tensor]] = {"input": [], "output": []}
    original_input = runtime.input_relay
    original_output = runtime.output_relay

    class RelayCapture:
        def __init__(self, name: str, relay: Any) -> None:
            self.name = name
            self.relay = relay

        def __call__(self, field: Any) -> Any:
            value = self.relay(field)
            _require(hasattr(value, "tensor") and torch.is_complex(value.tensor), f"{self.name} relay did not return a complex field")
            relays[self.name].append(value.tensor.detach().cpu().contiguous())
            return value

        def __getattr__(self, name: str) -> Any:
            return getattr(self.relay, name)

    runtime.input_relay = RelayCapture("input", original_input)
    runtime.output_relay = RelayCapture("output", original_output)
    try:
        result = runtime.forward_from_image(batch)
        _require(result.image_input is not None and result.ols_input_field is not None, "real-image runtime omitted its optical trace")
        _require(len(result.encoder_records) == 2 and len(result.generator_records) == 2, "runtime no longer has the expected two encoder and generator layers")
        _require(relays["input"] and relays["output"], "relay outputs were not captured")
        trace = {
            "slm_output": result.image_input.slm_field.tensor,
            "encoder_input": relays["input"][-1],
            "encoder_layer_1": result.encoder_records[0].tensor,
            "pre_ols": result.encoder_records[1].tensor,
            "post_ols": result.generator_input_field.tensor,
            "decoder_layer_1": result.generator_records[0].tensor,
            "decoder_layer_2": result.generator_records[1].tensor,
            "cmos_field": relays["output"][-1],
            "z_ols": result.z_ols.coefficients,
            "raw_intensity": result.raw_intensity,
            "target_intensity": result.image_input.slm_intensity,
        }
        for key, value in trace.items():
            _require(torch.is_tensor(value) and bool(torch.isfinite(value.real).all()) and (not torch.is_complex(value) or bool(torch.isfinite(value.imag).all())), f"invalid captured trace tensor: {key}")
        _require(tuple(trace["slm_output"].shape[-2:]) == (200, 200), "SLM plane geometry changed")
        _require(tuple(trace["encoder_input"].shape[-2:]) == (100, 100), "encoder input plane geometry changed")
        _require(tuple(trace["cmos_field"].shape[-2:]) == (200, 200), "CMOS incident field geometry changed")
        _require(trace["z_ols"].shape[-1] == 100 and torch.is_complex(trace["z_ols"]), "OLS 100-port complex state changed")
        _require(torch.allclose(trace["cmos_field"].abs().square(), trace["raw_intensity"].detach().cpu(), atol=1e-6, rtol=1e-6), "CMOS square-law readout mismatch")
        return trace
    finally:
        runtime.input_relay = original_input
        runtime.output_relay = original_output


def _evaluate_candidate(
    root: Path,
    consumer: Any,
    models: Mapping[str, Any],
    candidate: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    structural_mean: torch.Tensor,
    structural_scale: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any] | None]:
    consumer.load_trainable_state_dict(candidate["states"])
    consumer.runtime.eval()
    before = _state_digest({key: value for key, value in consumer.runtime.state_dict().items()})
    raws: list[torch.Tensor] = []
    logits: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    rec_losses: list[torch.Tensor] = []
    struct_losses: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, SAMPLE_COUNT, FORWARD_BATCH_SIZE):
            end = min(SAMPLE_COUNT, start + FORWARD_BATCH_SIZE)
            batch = images[start:end].to(device)
            result = consumer.runtime.forward_from_image(batch)
            raw = result.raw_intensity.detach()
            _require(raw.shape == (end - start, 1, 200, 200), f"raw CMOS geometry changed at step{candidate['step']}")
            _require(bool(torch.isfinite(raw).all()), f"raw CMOS contains nonfinite values at step{candidate['step']}")
            view = canonical_semantics.cmos_view(raw)
            for name, model in models.items():
                evaluation = model(view)
                prediction = evaluation[-1] if isinstance(evaluation, (tuple, list)) else evaluation
                logits[name].append(prediction.detach().cpu())
            target_view = canonical_semantics.cmos_view(batch)
            raw_struct = structural._normalized_surrogate(structural._structural_surrogate(view), structural_mean, structural_scale)
            target_struct = structural._normalized_surrogate(structural._structural_surrogate(target_view), structural_mean, structural_scale)
            struct_losses.append((raw_struct - target_struct).square().mean(dim=1).detach().cpu())
            rec_losses.append((raw - result.image_input.slm_intensity).square().flatten(1).mean(dim=1).detach().cpu())
            raws.append(raw.detach().cpu().contiguous())
    after = _state_digest({key: value for key, value in consumer.runtime.state_dict().items()})
    _require(before == after, f"step{candidate['step']} visualization forward changed runtime state")
    raw_all = torch.cat(raws, dim=0)
    subset_metrics: dict[str, Any] = {
        "rows": SAMPLE_COUNT,
        "role": "fixed 50-row DEV visualization subset; descriptive only",
        "E": {},
        "L_rec": float(torch.cat(rec_losses).mean()),
        "L_struct": float(torch.cat(struct_losses).mean()),
    }
    label_values = labels.cpu()
    for name in models:
        prediction = torch.cat(logits[name], dim=0).argmax(dim=1)
        recalls = []
        for digit in range(10):
            mask = label_values == digit
            recalls.append(float((prediction[mask] == label_values[mask]).float().mean()))
        subset_metrics["E"][name] = {
            "accuracy": float((prediction == label_values).float().mean()),
            "macro_recall": float(np.mean(recalls)),
            "per_class_recall": recalls,
        }
    full_dev_path = root / candidate["run_dir"] / f"dev_diagnostics_step_{candidate['step']:06d}.json"
    full_dev = _read_json(full_dev_path) if full_dev_path.is_file() else None
    return raw_all, {"subset_50": subset_metrics, "existing_full_dev": full_dev, "existing_full_dev_path": full_dev_path.relative_to(root).as_posix() if full_dev else None}, None


def _score_template(path: Path, aliases: Sequence[str], sample_ids: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_alias", "sample_id", "read_digit", "clear"])
        for alias in aliases:
            for sample_id in sample_ids:
                writer.writerow([alias, sample_id, "", ""])


def _render_blind_sheets(output_dir: Path, raw_by_alias: Mapping[str, torch.Tensor], aliases: Sequence[str], sample_ids: Sequence[str], vmax: float, seed: int) -> list[str]:
    order = np.random.default_rng(seed + 1).permutation(len(sample_ids)).tolist()
    page_paths: list[str] = []
    for page_index in range(math.ceil(len(order) / 5)):
        sample_rows = order[page_index * 5 : (page_index + 1) * 5]
        fig, axes = plt.subplots(len(sample_rows), len(aliases), figsize=(max(16.0, len(aliases) * 0.60), max(2.8, len(sample_rows) * 0.72)), squeeze=False)
        fig.subplots_adjust(left=0.025, right=0.995, top=0.88, bottom=0.08, wspace=0.02, hspace=0.12)
        for column, alias in enumerate(aliases):
            axes[0, column].set_title(alias, fontsize=5.3, pad=2)
        for row, sample_index in enumerate(sample_rows):
            for column, alias in enumerate(aliases):
                axis = axes[row, column]
                image = raw_by_alias[alias][sample_index, 0].numpy()
                axis.imshow(image, cmap="gray", vmin=0.0, vmax=vmax, origin="lower", interpolation="nearest")
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
            axes[row, 0].set_ylabel(sample_ids[sample_index], fontsize=6, rotation=0, ha="right", va="center", labelpad=19)
        fig.suptitle("Blind CMOS readability review · 200×200 raw intensity", fontsize=10, fontweight="bold")
        fig.text(0.5, 0.015, f"Shared display scale: 0–{vmax:.6g}; each column is one anonymous checkpoint.", ha="center", fontsize=6.5)
        path = output_dir / f"blind_cmos_readability_{page_index + 1:02d}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        page_paths.append(path.relative_to(ROOT).as_posix())
    pdf_path = output_dir / "blind_cmos_readability_sheets.pdf"
    with PdfPages(pdf_path) as pdf:
        for relative in page_paths:
            image = plt.imread(ROOT / relative)
            fig, axis = plt.subplots(figsize=(16, max(2.8, image.shape[0] / image.shape[1] * 16)))
            axis.imshow(image)
            axis.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return page_paths + [pdf_path.relative_to(ROOT).as_posix()]


def _prepare(root: Path, output_dir: Path, cuda_runtime: Mapping[str, Any]) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to overwrite existing visualization run: {output_dir}")
    registry = _candidate_registry(root)
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "candidate replay requires the project's CUDA environment")
    device = torch.device("cuda:0")
    authorities = robust._load_authorities(root)
    images, labels, source_indices, selection_manifest = _load_selection(root, authorities)
    consumer, models = _load_models(root, authorities, device)
    structural_mean, structural_scale, structural_normalization_verification = _load_structural_normalization(
        root, device, authorities["images"].index_select(0, authorities["train_indices"])
    )
    initial_frozen = robust._state_digest(robust._frozen_runtime_state(consumer))
    initial_full = _state_digest(consumer.runtime.state_dict())

    step_to_alias, alias_to_step = _ranked_aliases(registry)
    sample_order = np.random.default_rng(RATING_SEED + 2).permutation(SAMPLE_COUNT).tolist()
    sample_ids = [f"R{index + 1:03d}" for index in range(SAMPLE_COUNT)]
    sample_id_for_position = {position: sample_ids[rank] for rank, position in enumerate(sample_order)}
    ordered_sample_ids = [sample_id_for_position[position] for position in range(SAMPLE_COUNT)]
    raw_by_step: dict[int, torch.Tensor] = {}
    candidate_metrics: dict[int, Any] = {}
    source_state_hashes: dict[int, str] = {}
    for candidate in registry:
        raw, metrics, _ = _evaluate_candidate(root, consumer, models, candidate, images, labels, structural_mean, structural_scale, device)
        raw_by_step[candidate["step"]] = raw
        candidate_metrics[candidate["step"]] = metrics
        source_state_hashes[candidate["step"]] = _state_digest({"encoder_state": candidate["states"]["encoder_state"], "generator_state": candidate["states"]["generator_state"]})
    _require(initial_frozen == robust._state_digest(robust._frozen_runtime_state(consumer)), "visualization changed frozen optical state")
    raw_values = torch.cat([raw_by_step[step].flatten() for step in CANDIDATE_STEPS]).numpy()
    display_vmax = float(np.quantile(raw_values, 0.995))
    _require(math.isfinite(display_vmax) and display_vmax > 0.0, "shared CMOS display scale is invalid")
    output_dir.mkdir(parents=True, exist_ok=False)
    alias_order = [step_to_alias[candidate["step"]] for candidate in registry]
    np.random.default_rng(RATING_SEED + 3).shuffle(alias_order)
    raw_by_alias = {step_to_alias[step]: raw_by_step[step] for step in CANDIDATE_STEPS}
    torch.save({"schema": "cmos_step_comparison_candidate_raw.v1", "steps": list(CANDIDATE_STEPS), "raw_intensity": torch.stack([raw_by_step[step] for step in CANDIDATE_STEPS])}, output_dir / "candidate_cmos_raw.pt")
    _write_json(output_dir / UNBLINDING_KEY_NAME, {"schema": "blind_cmos_candidate_key.v1", "candidate_alias_to_step": alias_to_step, "rating_column_order": alias_order, "sample_id_to_source": {sample_id_for_position[position]: {"position_in_fixed_selection": position, "canonical_source_index": source_indices[position], "label": int(labels[position])} for position in range(SAMPLE_COUNT)}, "rating_order": "record each blind guess and clear flag before opening this file"})
    _write_json(output_dir / "checkpoint_manifest.json", {"schema": "cmos_step_comparison_checkpoint_manifest.v1", "frozen_max_step": 9300, "candidate_count": len(registry), "candidate_steps": list(CANDIDATE_STEPS), "candidates": [{key: value for key, value in candidate.items() if key != "states"} for candidate in registry], "step9300_status": "valid execution report; outer scientific condition failed; descriptive candidate only", "future_checkpoint_discovery": False})
    _write_json(output_dir / "sample_manifest.json", {"schema": "cmos_step_comparison_fixed_dev_selection.v1", "source_manifest_path": SELECTION_MANIFEST.as_posix(), "source_manifest_sha256": file_sha256(root / SELECTION_MANIFEST), "selection_source": SELECTION_SOURCE_INDICES, "count": SAMPLE_COUNT, "per_class": 5, "canonical_source_indices": source_indices, "blind_sample_ids": {sample_id_for_position[position]: {"position": position, "source_index": source_indices[position]} for position in range(SAMPLE_COUNT)}, "rating_order_seed": RATING_SEED, "target_labels_withheld_until_after_rating": True})
    _write_json(output_dir / "candidate_metrics_50_dev.json", {str(step): candidate_metrics[step] for step in CANDIDATE_STEPS})
    all_candidate_metric_rows: list[dict[str, Any]] = []
    for step in CANDIDATE_STEPS:
        values = candidate_metrics[step]
        subset = values["subset_50"]
        row: dict[str, Any] = {
            "step": step,
            "subset_rows": subset["rows"],
            "L_rec_50": subset["L_rec"],
            "L_struct_50": subset["L_struct"],
            "full_DEV_5000_source": values["existing_full_dev_path"],
            "full_DEV_5000_metrics": json.dumps(_full_dev_summary(values["existing_full_dev"]), ensure_ascii=False) if values["existing_full_dev"] else "unavailable for this stored step",
        }
        for model_name, metric in subset["E"].items():
            row[f"{model_name}_accuracy_50"] = metric["accuracy"]
            row[f"{model_name}_macro_recall_50"] = metric["macro_recall"]
        all_candidate_metric_rows.append(row)
    _write_csv(output_dir / "candidate_metrics_50_dev.csv", all_candidate_metric_rows)
    _score_template(output_dir / "readability_scores_template.csv", [step_to_alias[step] for step in CANDIDATE_STEPS], ordered_sample_ids)
    sheet_files = _render_blind_sheets(output_dir, raw_by_alias, alias_order, ordered_sample_ids, display_vmax, RATING_SEED)
    _write_json(output_dir / "visualization_metadata.json", {"schema": "cmos_eight_node_step_comparison_metadata.v1", "task_classification": "DESCRIPTIVE_VISUALIZATION_ONLY", "candidate_upper_bound": 9300, "candidate_steps": list(CANDIDATE_STEPS), "same_fixed_real_dev_samples_for_all_steps": True, "raw_cmos_shape": [SAMPLE_COUNT, 1, 200, 200], "shared_cmos_display_vmax_p995": display_vmax, "per_image_contrast_stretch": False, "models": {name: "Frozen 128D-feature/10-class electronic evaluator; evaluation only, never a propagation plane" for name in models}, "structural_normalization_verification": structural_normalization_verification, "cuda_runtime": dict(cuda_runtime), "runtime_state_sha256_before_candidate_loads": initial_full, "frozen_runtime_state_sha256_before": initial_frozen, "per_forward_runtime_integrity": "checked equal before/after each checkpoint replay and each finalist node trace", "blind_review_sheets": sheet_files, "score_template": "readability_scores_template.csv", "status": "AWAITING_BLIND_READABILITY_SCORES", "training_or_optimizer_or_backward_executed": False})
    return {"result": "AWAITING_BLIND_READABILITY_SCORES", "candidate_count": len(registry), "candidate_steps": list(CANDIDATE_STEPS), "output_dir": output_dir.relative_to(root).as_posix(), "blind_review_sheets": sheet_files, "score_template": (output_dir / "readability_scores_template.csv").relative_to(root).as_posix()}


def _reblind_existing(output_dir: Path) -> dict[str, Any]:
    """Replace the pending rating sheets with a fresh random alias and column order."""
    _require(output_dir.is_dir(), f"prepared visualization run is missing: {output_dir}")
    meta_path = output_dir / "visualization_metadata.json"
    meta = _read_json(meta_path)
    _require(meta.get("status") == "AWAITING_BLIND_READABILITY_SCORES", "only a pending visualization can be reblinded")
    raw_artifact = safe_torch_load(output_dir / "candidate_cmos_raw.pt", map_location="cpu")
    _require(isinstance(raw_artifact, Mapping) and raw_artifact.get("schema") == "cmos_step_comparison_candidate_raw.v1", "candidate CMOS observations are missing")
    steps = raw_artifact.get("steps")
    raw_stack = raw_artifact.get("raw_intensity")
    _require(steps == list(CANDIDATE_STEPS), "candidate CMOS observation order changed")
    _require(torch.is_tensor(raw_stack) and raw_stack.shape == (len(CANDIDATE_STEPS), SAMPLE_COUNT, 1, 200, 200), "candidate CMOS observation geometry changed")
    sample_manifest = _read_json(output_dir / "sample_manifest.json")
    existing_key_path = output_dir / UNBLINDING_KEY_NAME
    legacy_key_path = output_dir / "blind_step_key.json"
    existing_key = _read_json(existing_key_path if existing_key_path.is_file() else legacy_key_path)
    blind_samples = existing_key.get("sample_id_to_source")
    _require(isinstance(blind_samples, Mapping) and len(blind_samples) == SAMPLE_COUNT, "fixed sample identities are missing")

    randomized_steps = list(CANDIDATE_STEPS)
    secrets.SystemRandom().shuffle(randomized_steps)
    alias_to_step = {f"Q{index + 1:02d}": step for index, step in enumerate(randomized_steps)}
    step_to_alias = {step: alias for alias, step in alias_to_step.items()}
    aliases = list(alias_to_step)
    column_order = aliases.copy()
    secrets.SystemRandom().shuffle(column_order)
    sample_ids = list(blind_samples)
    _write_json(output_dir / UNBLINDING_KEY_NAME, {
        "schema": "blind_cmos_candidate_key.v1",
        "candidate_alias_to_step": alias_to_step,
        "rating_column_order": column_order,
        "sample_id_to_source": {
            sample_id: {
                "position_in_fixed_selection": int(value.get("position_in_fixed_selection", value.get("position"))),
                "canonical_source_index": int(value.get("canonical_source_index", value.get("source_index"))),
                "label": int(value["label"]),
            }
            for sample_id, value in blind_samples.items()
        },
        "rating_order": "record the visual digit guess and clear flag before unblinding this key",
    })
    sample_manifest["blind_sample_ids"] = {
        sample_id: {
            "position": int(value.get("position_in_fixed_selection", value.get("position"))),
            "source_index": int(value.get("canonical_source_index", value.get("source_index"))),
        }
        for sample_id, value in blind_samples.items()
    }
    sample_manifest.pop("labels", None)
    sample_manifest["target_labels_withheld_until_after_rating"] = True
    _write_json(output_dir / "sample_manifest.json", sample_manifest)
    if legacy_key_path.is_file():
        legacy_key_path.unlink()

    raw_by_alias = {
        step_to_alias[step]: raw_stack[index]
        for index, step in enumerate(CANDIDATE_STEPS)
    }
    sheet_files = _render_blind_sheets(
        output_dir,
        raw_by_alias,
        column_order,
        sample_ids,
        float(meta["shared_cmos_display_vmax_p995"]),
        secrets.randbelow(2**31),
    )
    _score_template(output_dir / "readability_scores_template.csv", aliases, sample_ids)
    meta.update({
        "status": "AWAITING_BLIND_READABILITY_SCORES",
        "blind_review_sheets": sheet_files,
        "blind_column_order_randomized": True,
        "blind_aliases_regenerated": True,
        "rating_pass": "fresh reblind of the same frozen 27-checkpoint, 50-sample observations",
    })
    _write_json(meta_path, meta)
    return {
        "result": "AWAITING_BLIND_READABILITY_SCORES",
        "candidate_count": len(CANDIDATE_STEPS),
        "sample_count": SAMPLE_COUNT,
        "blind_review_sheets": sheet_files,
        "score_template": (output_dir / "readability_scores_template.csv").relative_to(ROOT).as_posix(),
    }


def _load_scores(path: Path, aliases: Sequence[str], sample_ids: Sequence[str]) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(alias, sample_id) for alias in aliases for sample_id in sample_ids}
    observed: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    _require(len(rows) == len(expected), "readability score file does not contain exactly one row per candidate and sample")
    for row in rows:
        alias = str(row.get("candidate_alias", "")).strip()
        sample_id = str(row.get("sample_id", "")).strip()
        _require((alias, sample_id) in expected and (alias, sample_id) not in observed, "readability score has an unknown or duplicate candidate/sample pair")
        guess = str(row.get("read_digit", "")).strip()
        clear_text = str(row.get("clear", "")).strip()
        _require(guess in {"?", *(str(value) for value in range(10))}, f"invalid blind digit for {alias}/{sample_id}")
        _require(clear_text in {"0", "1"}, f"clear must be 0 or 1 for {alias}/{sample_id}")
        observed.add((alias, sample_id))
        normalized.append({"candidate_alias": alias, "sample_id": sample_id, "read_digit": guess, "clear": int(clear_text)})
    _require(observed == expected, "readability score file is incomplete")
    return normalized


def _readability_rank(scores: Sequence[Mapping[str, Any]], key: Mapping[str, Any]) -> list[dict[str, Any]]:
    sample_labels = {str(name): int(value["label"]) for name, value in key["sample_id_to_source"].items()}
    aliases = key["candidate_alias_to_step"]
    counts = {alias: {"correct_clear": 0, "correct": 0, "clear": 0} for alias in aliases}
    class_counts = {alias: {digit: {"correct_clear": 0, "correct": 0, "clear": 0, "n": 0} for digit in range(10)} for alias in aliases}
    for row in scores:
        alias = str(row["candidate_alias"])
        digit = sample_labels[str(row["sample_id"])]
        correct = row["read_digit"] == str(digit)
        clear = int(row["clear"]) == 1
        counts[alias]["correct"] += int(correct)
        counts[alias]["clear"] += int(clear)
        counts[alias]["correct_clear"] += int(correct and clear)
        class_counts[alias][digit]["n"] += 1
        class_counts[alias][digit]["correct"] += int(correct)
        class_counts[alias][digit]["clear"] += int(clear)
        class_counts[alias][digit]["correct_clear"] += int(correct and clear)
    ranked = []
    for alias, step in aliases.items():
        class_columns: dict[str, int] = {}
        for digit in range(10):
            values = class_counts[alias][digit]
            class_columns[f"digit_{digit}_correct_clear"] = values["correct_clear"]
            class_columns[f"digit_{digit}_correct"] = values["correct"]
            class_columns[f"digit_{digit}_clear"] = values["clear"]
        ranked.append({"candidate_alias": alias, "step": int(step), **counts[alias], **class_columns, "n": SAMPLE_COUNT, "correct_clear_rate": counts[alias]["correct_clear"] / SAMPLE_COUNT, "correct_rate": counts[alias]["correct"] / SAMPLE_COUNT, "clear_rate": counts[alias]["clear"] / SAMPLE_COUNT})
    ranked.sort(key=lambda row: (-row["correct_clear"], -row["correct"], row["step"]))
    for rank, row in enumerate(ranked, start=1):
        row["readability_rank"] = rank
        row["selection_role"] = "best" if rank == 1 else "worst" if rank == len(ranked) else "median" if rank == (len(ranked) + 1) // 2 else None
    return ranked


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise RuntimeError("cannot write a CSV without rows or declared columns")
    columns = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _capture_selected_nodes(root: Path, candidate: Mapping[str, Any], consumer: Any, models: Mapping[str, Any], display_images: torch.Tensor, display_labels: torch.Tensor, structural_mean: torch.Tensor, structural_scale: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    consumer.load_trainable_state_dict(candidate["states"])
    consumer.runtime.eval()
    before = _state_digest(consumer.runtime.state_dict())
    captures: dict[str, list[torch.Tensor]] = {key: [] for key in (*NODE_KEYS, "z_ols", "raw_intensity", "target_intensity")}
    for start in range(0, len(display_images), FORWARD_BATCH_SIZE):
        batch = display_images[start : start + FORWARD_BATCH_SIZE].to(device)
        with torch.inference_mode():
            trace = _capture_batch(consumer, batch)
        for name in captures:
            captures[name].append(trace[name].detach().cpu().contiguous())
    after = _state_digest(consumer.runtime.state_dict())
    _require(before == after, f"step{candidate['step']} node capture mutated runtime state")
    result = {name: torch.cat(values, dim=0) for name, values in captures.items()}
    result["runtime_state_sha256_before"] = before
    result["runtime_state_sha256_after"] = after
    return result


def _figure_style() -> None:
    matplotlib.rcParams.update({"font.family": ["Arial", "Microsoft YaHei", "DejaVu Sans", "sans-serif"], "font.size": 8, "axes.titlesize": 8, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def _save_figure(fig: plt.Figure, base: Path) -> list[str]:
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    svg = base.with_suffix(".svg")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png.relative_to(ROOT).as_posix(), pdf.relative_to(ROOT).as_posix(), svg.relative_to(ROOT).as_posix()]


def _write_alignment_manifest(fig: plt.Figure, axes: Sequence[Any], base: Path) -> Path:
    """Record final Matplotlib plot rectangles for the alignment auditor."""
    fig.canvas.draw()
    width_in, height_in = (float(value) for value in fig.get_size_inches())
    width_pt, height_pt = width_in * 72.0, height_in * 72.0
    panels: list[dict[str, Any]] = []
    for index, axis in enumerate(axes):
        subplot_spec = axis.get_subplotspec()
        _require(subplot_spec is not None, "comparison panel has no Matplotlib grid geometry")
        position = axis.get_position(original=False)
        panels.append({
            "id": f"panel-{index + 1:03d}",
            "bbox_pt": [position.x0 * width_pt, position.y0 * height_pt, position.x1 * width_pt, position.y1 * height_pt],
            "grid_id": "main-grid",
            "row_start": int(subplot_spec.rowspan.start),
            "row_stop": int(subplot_spec.rowspan.stop),
            "col_start": int(subplot_spec.colspan.start),
            "col_stop": int(subplot_spec.colspan.stop),
        })
    manifest = {
        "schema_version": 1,
        "backend": "python-matplotlib",
        "figure": {"width_pt": width_pt, "height_pt": height_pt},
        "panels": panels,
        "exemptions": [],
    }
    path = base.with_name(base.name + ".alignment-layout.json")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _field_amplitude(field: torch.Tensor) -> np.ndarray:
    _require(torch.is_complex(field) and field.ndim == 4 and field.shape[1] == 1, "amplitude renderer requires a complex [N,1,H,W] field")
    return field.detach().abs().cpu().numpy()[:, 0]


def _field_phase(field: torch.Tensor) -> np.ndarray:
    _require(torch.is_complex(field) and field.ndim == 4 and field.shape[1] == 1, "phase renderer requires a complex [N,1,H,W] field")
    return torch.remainder(torch.angle(field.detach()), PI2).cpu().numpy()[:, 0]


def _add_ols_inset(axis: Any, z: torch.Tensor, sample_index: int, mode: str, amplitude_scale: float) -> None:
    matrix = z.detach().cpu().reshape(-1, 10, 10)[sample_index]
    inset = inset_axes(axis, width="28%", height="28%", loc="lower right", borderpad=0.45)
    if mode == "amplitude":
        inset.imshow((matrix.abs().numpy() / amplitude_scale).clip(0.0, 1.0), cmap="magma", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
    else:
        phase = torch.remainder(torch.angle(matrix), PI2).numpy()
        magnitude = matrix.abs().numpy()
        inset.imshow(np.ma.masked_where(magnitude < amplitude_scale * 0.01, phase), cmap="twilight", vmin=0.0, vmax=PI2, origin="lower", interpolation="nearest")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.4)


def _render_trace(output_dir: Path, step: int, trace: Mapping[str, Any], amplitudes: Mapping[str, np.ndarray], phases: Mapping[str, np.ndarray], amplitude_scales: Mapping[str, float], z_scale: float, labels: torch.Tensor) -> list[str]:
    results: list[str] = []
    for mode in ("amplitude", "phase"):
        fig, axes = plt.subplots(DISPLAY_PER_CLASS * 10, len(NODE_KEYS), figsize=(17.2, 27.0), squeeze=False)
        fig.subplots_adjust(left=0.055, right=0.965, top=0.935, bottom=0.055, wspace=0.045, hspace=0.12)
        last_image = None
        for row in range(len(labels)):
            for column, key in enumerate(NODE_KEYS):
                axis = axes[row, column]
                if mode == "amplitude":
                    last_image = axis.imshow(np.clip(amplitudes[key][row] / amplitude_scales[key], 0.0, 1.0), cmap="magma", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest", aspect="equal")
                    if column == 4:
                        _add_ols_inset(axis, trace["z_ols"], row, mode, z_scale)
                else:
                    last_image = axis.imshow(phases[key][row], cmap="twilight", vmin=0.0, vmax=PI2, origin="lower", interpolation="nearest", aspect="equal")
                    if column == 4:
                        _add_ols_inset(axis, trace["z_ols"], row, mode, z_scale)
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(NODE_LABELS[column], fontsize=7.2, pad=5)
                if column == 0:
                    axis.set_ylabel(f"y={int(labels[row])} · #{row % DISPLAY_PER_CLASS + 1}", fontsize=6.3, rotation=0, ha="right", va="center", labelpad=11)
                for spine in axis.spines.values():
                    spine.set_visible(False)
        _require(last_image is not None, "eight-node trace figure is empty")
        if mode == "amplitude":
            scalar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="magma")
            scalar.set_array([])
            cbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), fraction=0.008, pad=0.006)
            cbar.set_label("amplitude / fixed node-wise 99.5th percentile across three selected steps", fontsize=7)
            title = f"step{step} real-path optical trace · amplitude"
        else:
            scalar = ScalarMappable(norm=Normalize(0.0, PI2), cmap="twilight")
            scalar.set_array([])
            cbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), fraction=0.008, pad=0.006, ticks=[0, math.pi / 2, math.pi, 3 * math.pi / 2, PI2])
            cbar.ax.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
            cbar.set_label("actual wrapped phase of complex field (rad), [0, 2π)", fontsize=7)
            title = f"step{step} real-path optical trace · wrapped phase"
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.966)
        fig.text(0.055, 0.027, "Rows: fixed real DEV examples, two per digit. OLS inset: row-major 10×10 complex coefficients. Node shapes are native resolution; CMOS intensity is |incident field|².", fontsize=7.0, ha="left")
        base = output_dir / f"step{step:06d}_{mode}"
        _write_alignment_manifest(fig, axes.ravel().tolist(), base)
        results.extend(_save_figure(fig, base))
    return results


def _render_finalist_cmos(output_dir: Path, finalists: Sequence[Mapping[str, Any]], raw_by_step: Mapping[int, torch.Tensor], fixed_targets: torch.Tensor, sample_positions: Sequence[int], labels: torch.Tensor, vmax: float) -> list[str]:
    fig, axes = plt.subplots(len(sample_positions), len(finalists) + 1, figsize=(7.4, 16.0), squeeze=False)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.96, bottom=0.035, wspace=0.05, hspace=0.06)
    axes[0, 0].set_title("DEV target", fontsize=8)
    for column, item in enumerate(finalists, start=1):
        axes[0, column].set_title(f"{item['selection_role']} · step{item['step']}", fontsize=8)
    for row, source_pos in enumerate(sample_positions):
        for column in range(len(finalists) + 1):
            axis = axes[row, column]
            if column == 0:
                target = fixed_targets[row, 0].numpy()
            else:
                target = raw_by_step[int(finalists[column - 1]["step"])][source_pos, 0].numpy()
            axis.imshow(target, cmap="gray", vmin=0.0, vmax=vmax, origin="lower", interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if column == 0:
                axis.set_ylabel(f"y={int(labels[row])} · #{row % 2 + 1}", rotation=0, ha="right", va="center", labelpad=12, fontsize=6)
    base = output_dir / "finalist_cmos_comparison"
    _write_alignment_manifest(fig, axes.ravel().tolist(), base)
    return _save_figure(fig, base)


def _full_dev_summary(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    metrics: dict[str, Any] = {"rows": value.get("rows", 5000), "step": value.get("step")}
    for name in ("E_S", "E_I", "E_C", "E_D"):
        entry = value.get(name)
        if isinstance(entry, Mapping):
            metrics[name] = {key: entry.get(key) for key in ("accuracy", "macro_recall") if key in entry}
    for name in ("L_rec", "L_struct", "L_op"):
        if name in value:
            metrics[name] = value[name]
    return metrics


def _finalize(root: Path, output_dir: Path, score_path: Path) -> dict[str, Any]:
    _require(output_dir.is_dir(), f"prepared visualization run is missing: {output_dir}")
    meta = _read_json(output_dir / "visualization_metadata.json")
    _require(meta.get("status") == "AWAITING_BLIND_READABILITY_SCORES", "visualization run is already finalized or has invalid status")
    key = _read_json(output_dir / UNBLINDING_KEY_NAME)
    aliases = list(key["candidate_alias_to_step"])
    sample_manifest = _read_json(output_dir / "sample_manifest.json")
    sample_ids = list(sample_manifest["blind_sample_ids"])
    scores = _load_scores(score_path, aliases, sample_ids)
    ranked = _readability_rank(scores, key)
    roles = {"best": ranked[0], "median": ranked[(len(ranked) - 1) // 2], "worst": ranked[-1]}
    for role, item in roles.items():
        item["selection_role"] = role
    _write_csv(output_dir / "readability_scores.csv", scores)
    _write_csv(output_dir / "readability_ranking_27_candidates.csv", ranked)
    _write_json(output_dir / "selected_steps.json", {"schema": "cmos_readability_selected_steps.v1", "ranking_rule": "correct and clear count descending; correct count descending; lower step first", "median_rule": "14th of 27 one-indexed", "selection_basis": "blind visual reading of fixed 50 real DEV CMOS outputs", "selected": roles})

    candidates = _candidate_registry(root)
    candidate_by_step = {item["step"]: item for item in candidates}
    authorities = robust._load_authorities(root)
    all_images, all_labels, source_indices, _selection = _load_selection(root, authorities)
    display_positions = []
    class_seen = [0] * 10
    for index, label in enumerate(all_labels.tolist()):
        if class_seen[label] < DISPLAY_PER_CLASS:
            display_positions.append(index)
            class_seen[label] += 1
    display_images = all_images[display_positions].contiguous()
    display_labels = all_labels[display_positions].contiguous()
    _require(display_labels.tolist() == [digit for digit in range(10) for _ in range(DISPLAY_PER_CLASS)], "final display rows are not the first two samples per class")
    consumer, models = _load_models(root, authorities, torch.device("cuda:0"))
    structural_mean, structural_scale, _normalization_check = _load_structural_normalization(
        root, torch.device("cuda:0"), authorities["images"].index_select(0, authorities["train_indices"])
    )
    traces: dict[int, dict[str, Any]] = {}
    for role in ("best", "median", "worst"):
        step = int(roles[role]["step"])
        traces[step] = _capture_selected_nodes(root, candidate_by_step[step], consumer, models, display_images, display_labels, structural_mean, structural_scale, torch.device("cuda:0"))

    amplitude_scales: dict[str, float] = {}
    phase_views: dict[int, dict[str, np.ndarray]] = {}
    amplitude_views: dict[int, dict[str, np.ndarray]] = {}
    z_scale = 0.0
    for key_name in NODE_KEYS:
        magnitudes = torch.cat([traces[step][key_name].abs().flatten() for step in traces]).numpy()
        amplitude_scales[key_name] = float(np.quantile(magnitudes, 0.995))
        _require(amplitude_scales[key_name] > 0.0 and math.isfinite(amplitude_scales[key_name]), f"invalid shared amplitude scale for {key_name}")
    z_all = torch.cat([traces[step]["z_ols"].abs().flatten() for step in traces]).numpy()
    z_scale = float(np.quantile(z_all, 0.995))
    for step, trace in traces.items():
        amplitude_views[step] = {node: _field_amplitude(trace[node]) for node in NODE_KEYS}
        phase_views[step] = {node: _field_phase(trace[node]) for node in NODE_KEYS}
    figure_files: list[str] = []
    for role in ("best", "median", "worst"):
        step = int(roles[role]["step"])
        figure_files.extend(_render_trace(output_dir, step, traces[step], amplitude_views[step], phase_views[step], amplitude_scales, z_scale, display_labels))

    raw_artifact = safe_torch_load(output_dir / "candidate_cmos_raw.pt", map_location="cpu")
    _require(isinstance(raw_artifact, Mapping) and raw_artifact.get("schema") == "cmos_step_comparison_candidate_raw.v1", "candidate CMOS observations are missing")
    _require(raw_artifact.get("steps") == list(CANDIDATE_STEPS), "candidate CMOS observation order changed")
    raw_stack = raw_artifact.get("raw_intensity")
    _require(torch.is_tensor(raw_stack) and raw_stack.shape == (len(CANDIDATE_STEPS), SAMPLE_COUNT, 1, 200, 200), "candidate CMOS observation geometry changed")
    raw_by_step = {step: raw_stack[index] for index, step in enumerate(CANDIDATE_STEPS)}
    top_candidates = [roles[name] for name in ("best", "median", "worst")]
    vmax = float(meta["shared_cmos_display_vmax_p995"])
    figure_files.extend(_render_finalist_cmos(output_dir, top_candidates, raw_by_step, display_images, display_positions, display_labels, vmax))

    metrics_50 = _read_json(output_dir / "candidate_metrics_50_dev.json")
    selected_metric_rows: list[dict[str, Any]] = []
    for role in ("best", "median", "worst"):
        step = int(roles[role]["step"])
        candidate_metrics = metrics_50[str(step)]
        record: dict[str, Any] = {"role": role, "step": step, "readability_rank": roles[role]["readability_rank"], "correct_clear_count_50": roles[role]["correct_clear"], "correct_count_50": roles[role]["correct"], "clear_count_50": roles[role]["clear"], "L_rec_50": candidate_metrics["subset_50"]["L_rec"], "L_struct_50": candidate_metrics["subset_50"]["L_struct"]}
        for model_name, values in candidate_metrics["subset_50"]["E"].items():
            record[f"{model_name}_accuracy_50"] = values["accuracy"]
            record[f"{model_name}_macro_recall_50"] = values["macro_recall"]
        record["full_DEV_5000_source"] = candidate_metrics["existing_full_dev_path"]
        record["full_DEV_5000_metrics"] = json.dumps(_full_dev_summary(candidate_metrics["existing_full_dev"]), ensure_ascii=False) if candidate_metrics["existing_full_dev"] else "unavailable for this stored step"
        selected_metric_rows.append(record)
    _write_csv(output_dir / "selected_step_metrics.csv", selected_metric_rows)

    node_registry = []
    for index, (key_name, label) in enumerate(zip(NODE_KEYS, NODE_LABELS, strict=True), start=1):
        first = traces[int(roles["best"]["step"])][key_name]
        node_registry.append({"node_index": index, "key": key_name, "label": label, "shape": list(first.shape), "dtype": str(first.dtype), "quantity": "amplitude |E| and wrapped phase arg(E)"})
    _write_json(output_dir / "node_registry.json", {"schema": "cmos_eight_node_field_registry.v1", "nodes": node_registry, "auxiliary_fields": [{"key": "z_ols", "label": "OLS 100-port complex coefficients", "shape": [len(display_labels), 100], "dtype": str(traces[int(roles["best"]["step"])]["z_ols"].dtype), "layout": "row-major 10×10 complex matrix per sample"}], "phase_policy": "actual angle from each captured complex field, wrapped to [0,2π); no phase is inferred from intensity", "camera_readout": "|cmos_field|² equals raw_intensity"})
    _write_json(output_dir / "visualization_metadata.json", {**meta, "status": "VALID_COMPLETED_DESCRIPTIVE_VISUALIZATION", "selected_steps": roles, "rating_method": "single AI visual reviewer; blinded per-image digit guess plus clear flag on the shared raw-CMOS scale; step/class key opened only after the score CSV was saved", "ranked_candidates": len(ranked), "figures": figure_files, "shared_node_amplitude_scales_p995": amplitude_scales, "shared_ols_amplitude_scale_p995": z_scale, "selected_trace_state_hashes": {str(step): {"before": trace["runtime_state_sha256_before"], "after": trace["runtime_state_sha256_after"]} for step, trace in traces.items()}, "training_or_optimizer_or_backward_executed": False})
    (output_dir / "README.md").write_text(
        "# CMOS readability and eight-node field comparison\n\n"
        "This is a descriptive comparison of the fixed 27-checkpoint allowlist through step9300 on the same 50 real DEV images (five per class). Later checkpoints are excluded. Best, median, and worst refer only to this sample-level CMOS readability ranking; they are not hardware validation, acceptance, or checkpoint promotion. Step9300 belongs to a canary whose outer scientific condition failed, so it is included as a descriptive candidate only.\n\n"
        "Ratings were made by one AI visual reviewer, with candidate aliases blinded and the same raw-CMOS intensity scale used throughout. Each candidate/sample pair records a digit guess and a binary clear flag; an unreadable tile is recorded as ?. This is a single-reviewer descriptive assessment, not an independent human consensus. Ranking uses correct-and-clear count, then correct count, then earlier step; the median is rank 14 of 27. Electronic evaluator scores and reconstruction/structure metrics are explanatory columns and do not select the three steps. Existing metrics over 5000 DEV rows are identified separately from metrics recomputed over this 50-row subset.\n\n"
        "The eight-node panels show the actual complex field amplitude and wrapped phase at each optical plane. The OLS inset shows the actual complex 100-port coefficients in row-major 10×10 form. CMOS intensity is |CMOS incident field|², checked against the stored raw camera-plane intensity. Shared display scales are used; no per-image contrast stretching is applied.\n\n"
        "No training, backward pass, optimizer update, or checkpoint write was executed. Candidate and finalist replays checked runtime state equality before and after inference.\n",
        encoding="utf-8",
    )
    return {"result": "VALID_COMPLETED_DESCRIPTIVE_VISUALIZATION", "output_dir": output_dir.relative_to(root).as_posix(), "selected_steps": {role: value["step"] for role, value in roles.items()}, "figure_count": len(figure_files), "score_rows": len(scores)}


def _capture_m0_endpoint(root: Path, endpoint: Mapping[str, Any], consumer: Any, images: torch.Tensor, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint_path = root / endpoint["checkpoint"]
    status_path = root / endpoint["status_path"]
    payload = safe_torch_load(checkpoint_path, map_location="cpu")
    attempt_status = _read_json(status_path)
    _require(attempt_status.get("state") == endpoint["parent_state"], f"M0 parent attempt status changed: {endpoint['key']}")
    _require(isinstance(payload, Mapping), f"M0 checkpoint is malformed: {endpoint['key']}")
    _require(payload.get("schema") == endpoint["schema"], f"M0 checkpoint schema changed: {endpoint['key']}")
    _require(payload.get("step") == 1200, f"M0 endpoint step changed: {endpoint['key']}")
    state = payload.get("trainable_parameter_state")
    _require(isinstance(state, Mapping) and set(state) == {"encoder.phase", "generator.phase"}, f"M0 trainable state changed: {endpoint['key']}")
    for name, expected_shape in (("encoder.phase", (2, 1, 100, 100)), ("generator.phase", (2, 1, 100, 100))):
        tensor = state[name]
        _require(torch.is_tensor(tensor) and tuple(tensor.shape) == expected_shape and tensor.dtype == torch.float32, f"invalid {name} tensor: {endpoint['key']}")
        _require(bool(torch.isfinite(tensor).all()), f"nonfinite {name} tensor: {endpoint['key']}")
    consumer.load_trainable_state_dict(state)
    consumer.runtime.eval()
    before = _state_digest(consumer.runtime.state_dict())
    captures: dict[str, list[torch.Tensor]] = {key: [] for key in (*NODE_KEYS, "z_ols", "raw_intensity")}
    for start in range(0, len(images), FORWARD_BATCH_SIZE):
        batch = images[start : start + FORWARD_BATCH_SIZE].to(device)
        with torch.inference_mode():
            trace = _capture_batch(consumer, batch)
        for name in captures:
            captures[name].append(trace[name].detach().cpu().contiguous())
    after = _state_digest(consumer.runtime.state_dict())
    _require(before == after, f"M0 endpoint forward changed runtime state: {endpoint['key']}")
    trace_out = {name: torch.cat(values, dim=0) for name, values in captures.items()}
    for name in NODE_KEYS:
        _require(torch.is_complex(trace_out[name]) and bool(torch.isfinite(trace_out[name].real).all()) and bool(torch.isfinite(trace_out[name].imag).all()), f"invalid M0 complex node: {endpoint['key']}/{name}")
    _require(torch.is_complex(trace_out["z_ols"]) and trace_out["z_ols"].shape == (len(images), 100), f"M0 OLS geometry changed: {endpoint['key']}")
    metadata = {
        "key": endpoint["key"],
        "label": endpoint["label"],
        "checkpoint": endpoint["checkpoint"].as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_schema": payload["schema"],
        "parent_attempt_status_path": endpoint["status_path"].as_posix(),
        "parent_attempt_status_sha256": file_sha256(status_path),
        "parent_attempt_state": attempt_status["state"],
        "step": payload["step"],
        "completed_updates": payload.get("completed_updates"),
        "evidence_status": endpoint["status"],
        "runtime_state_sha256_before": before,
        "runtime_state_sha256_after": after,
        "node_shapes": {name: list(trace_out[name].shape) for name in NODE_KEYS},
    }
    return trace_out, metadata


def _load_cloud_trace_model(root: Path, device: torch.device) -> tuple[Any, Mapping[str, Any], dict[str, Any]]:
    checkpoint_path = root / CLOUD_CHECKPOINT
    payload = safe_torch_load(checkpoint_path, map_location="cpu")
    _require(isinstance(payload, Mapping) and payload.get("schema") == "lightgen_400_mnist_semantic_baseline_checkpoint.v1", "cloud checkpoint schema changed")
    _require(payload.get("completed_updates") == 600, "cloud endpoint is not the saved step-600 checkpoint")
    state = payload.get("model")
    _require(isinstance(state, Mapping), "cloud checkpoint model state is missing")
    _require(tuple(state["encoder.phase"].shape) == (2, 1, 400, 400), "cloud encoder phase geometry changed")
    _require(tuple(state["decoder.phase"].shape) == (4, 1, 400, 400), "cloud decoder phase geometry changed")
    config = load_config(root / CLOUD_CONFIG)
    _require(config.system.kind == "diagnostic_lightgen_amplitude_fiber_ols_400", "cloud visualization config changed")
    model = LightGenAmplitudeFiberOLSSystemV2(config.system).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    report_path = checkpoint_path.parent / "report.json"
    report = _read_json(report_path)
    _require(report.get("schema") == "lightgen_400_mnist_semantic_baseline_report.v1", "cloud source report schema changed")
    _require(report.get("status") == "COMPLETED" and report.get("evidence_tier") == "diagnostic-only", "cloud evidence status changed")
    metadata = {
        "key": "cloud_large_step600",
        "label": "云端大结构 step600",
        "checkpoint": CLOUD_CHECKPOINT.as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_schema": payload["schema"],
        "completed_updates": payload["completed_updates"],
        "evidence_tier": report.get("evidence_tier"),
        "reported_status": report.get("status"),
        "source_report": CLOUD_CHECKPOINT.parent.as_posix() + "/report.json",
        "source_report_sha256": file_sha256(report_path),
        "reported_gate_passed": report.get("gate", {}).get("passed"),
        "remote_transfer_mapping": "user-provided AutoDL attempt_01 transfer; provider job identity not independently queried",
        "config": CLOUD_CONFIG.as_posix(),
        "config_sha256": file_sha256(root / CLOUD_CONFIG),
    }
    return model, payload, metadata


def _verify_cloud_generated_ols_replay(root: Path, model: Any, device: torch.device) -> dict[str, Any]:
    ols_path = root / CLOUD_GENERATED_OLS
    visuals_path = root / CLOUD_GENERATED_VISUALS
    ols = safe_torch_load(ols_path, map_location="cpu")
    visuals = safe_torch_load(visuals_path, map_location="cpu")
    _require(isinstance(ols, Mapping) and ols.get("schema") == "lightgen_400_mnist_generated_ols.v1", "cloud generated OLS schema changed")
    _require(isinstance(visuals, Mapping) and visuals.get("schema") == "lightgen_400_mnist_generated_visuals.v1", "cloud generated CMOS schema changed")
    coefficients = ols.get("coefficients")
    requested = ols.get("requested_labels")
    raw_examples = visuals.get("raw_cmos_examples_400x400")
    visual_labels = visuals.get("requested_labels")
    _require(torch.is_tensor(coefficients) and coefficients.shape == (2000, 100) and coefficients.dtype == torch.complex64, "cloud generated OLS tensor geometry changed")
    _require(torch.is_tensor(requested) and requested.shape == (2000,) and requested.dtype == torch.long, "cloud generated OLS labels are missing")
    _require(torch.bincount(requested, minlength=10).tolist() == [200] * 10, "cloud generated OLS class balance changed")
    _require(torch.is_tensor(raw_examples) and raw_examples.shape == (50, 1, 400, 400), "cloud saved CMOS sample geometry changed")
    _require(torch.is_tensor(visual_labels) and torch.equal(visual_labels, requested), "cloud generated OLS and CMOS label arrays differ")
    sample_positions = torch.tensor([digit * 200 + offset for digit in range(10) for offset in range(5)], dtype=torch.long)
    expected_labels = torch.tensor([digit for digit in range(10) for _ in range(5)], dtype=torch.long)
    _require(torch.equal(requested.index_select(0, sample_positions), expected_labels), "saved CMOS example ordering no longer matches the first five generated rows per class")
    _require(bool(torch.isfinite(coefficients.real).all()) and bool(torch.isfinite(coefficients.imag).all()), "cloud generated OLS contains nonfinite coefficients")
    _require(bool(torch.isfinite(raw_examples).all()), "cloud saved CMOS examples contain nonfinite values")

    before = _state_digest(model.state_dict())
    max_abs_error = 0.0
    absolute_error_sum = 0.0
    value_count = 0
    with torch.inference_mode():
        for start in range(0, len(sample_positions), 1):
            source_position = sample_positions[start : start + 1]
            z = coefficients.index_select(0, source_position).to(device)
            expected = raw_examples[start : start + 1].to(device)
            observed = model.forward_from_ols(z).raw_camera.tensor
            _require(torch.allclose(observed, expected, atol=1.0e-5, rtol=1.0e-4), f"cloud OLS replay differs from its saved CMOS sample at row {start}")
            difference = (observed - expected).abs()
            max_abs_error = max(max_abs_error, float(difference.max()))
            absolute_error_sum += float(difference.sum())
            value_count += difference.numel()
    after = _state_digest(model.state_dict())
    _require(before == after, "cloud generated OLS replay changed model state")
    return {
        "ols_path": CLOUD_GENERATED_OLS.as_posix(),
        "ols_sha256": file_sha256(ols_path),
        "saved_cmos_path": CLOUD_GENERATED_VISUALS.as_posix(),
        "saved_cmos_sha256": file_sha256(visuals_path),
        "replayed_rows": int(len(sample_positions)),
        "selection_rule": "first five stored generated examples per requested digit, rows 0..4 of each 200-row class block",
        "allclose_atol": 1.0e-5,
        "allclose_rtol": 1.0e-4,
        "max_absolute_error": max_abs_error,
        "mean_absolute_error": absolute_error_sum / value_count,
        "runtime_state_unchanged": before == after,
    }


def _capture_cloud_endpoint(model: Any, images: torch.Tensor, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    before = _state_digest(model.state_dict())
    captures: dict[str, list[torch.Tensor]] = {key: [] for key in (*CLOUD_NODE_KEYS, "z_ols", "raw_intensity")}
    for start in range(0, len(images), FORWARD_BATCH_SIZE):
        end = min(len(images), start + FORWARD_BATCH_SIZE)
        indices = torch.arange(start, end, dtype=torch.long)
        amplitude, _target = cloud_baseline._input_and_target(images, indices, device)
        with torch.inference_mode():
            result = model(model.command(amplitude))
        _require(len(result.optical_trace) == len(CLOUD_NODE_KEYS) == 10, "cloud trace no longer exposes the declared ten native nodes")
        fields = [item.tensor for item in result.optical_trace]
        for key, field in zip(CLOUD_NODE_KEYS, fields, strict=True):
            _require(torch.is_tensor(field) and torch.is_complex(field) and tuple(field.shape[1:]) == (1, 400, 400), f"invalid cloud complex node: {key}")
            _require(bool(torch.isfinite(field.real).all()) and bool(torch.isfinite(field.imag).all()), f"nonfinite cloud complex node: {key}")
            captures[key].append(field.detach().cpu().contiguous())
        coefficients = result.ols.coefficients
        raw = result.raw_camera.tensor
        _require(torch.is_complex(coefficients) and coefficients.shape == (end - start, 100), "cloud forward OLS geometry changed")
        _require(torch.allclose(fields[-1].abs().square(), raw, atol=1.0e-6, rtol=1.0e-6), "cloud CMOS intensity is not the square-law readout of its final complex field")
        captures["z_ols"].append(coefficients.detach().cpu().contiguous())
        captures["raw_intensity"].append(raw.detach().cpu().contiguous())
    after = _state_digest(model.state_dict())
    _require(before == after, "cloud real-image trace changed model state")
    trace_out = {name: torch.cat(values, dim=0) for name, values in captures.items()}
    metadata = {
        "node_count": len(CLOUD_NODE_KEYS),
        "node_shapes": {name: list(trace_out[name].shape) for name in CLOUD_NODE_KEYS},
        "plane_ids": ["lightgen_slm", "lightgen_encoder_layer_1", "lightgen_encoder_layer_2", "lightgen_fiber_input", "lightgen_fiber_reemission", "lightgen_generator_layer_1", "lightgen_generator_layer_2", "lightgen_generator_layer_3", "lightgen_generator_layer_4", "lightgen_raw_sensor"],
        "runtime_state_sha256_before": before,
        "runtime_state_sha256_after": after,
        "trace_input": "same fixed real DEV image examples as the two M0 endpoints; resized by the cloud training input transform to 400x400 and square-root encoded as amplitude",
    }
    return trace_out, metadata


def _render_endpoint_trace(
    output_dir: Path,
    stem: str,
    endpoint_label: str,
    trace: Mapping[str, torch.Tensor],
    node_keys: Sequence[str],
    node_labels: Sequence[str],
    amplitude_scales: Mapping[str, float],
    z_scale: float,
    labels: torch.Tensor,
    scale_note: str,
    endpoint_note: str,
) -> list[str]:
    from audit_panel_alignment import require_matplotlib_panel_alignment

    _require(len(node_keys) == len(node_labels), "node labels do not match the captured architecture")
    results: list[str] = []
    width = 17.2 if len(node_keys) == 8 else 20.0
    for mode in ("amplitude", "phase"):
        fig, axes = plt.subplots(len(labels), len(node_keys), figsize=(width, 27.0), squeeze=False)
        fig.subplots_adjust(left=0.065, right=0.955, top=0.935, bottom=0.065, wspace=0.045, hspace=0.12)
        field_views = {
            key: _field_amplitude(trace[key]) if mode == "amplitude" else np.ma.masked_where(
                _field_amplitude(trace[key]) < amplitude_scales[key] * 0.01,
                _field_phase(trace[key]),
            )
            for key in node_keys
        }
        last_image = None
        for row in range(len(labels)):
            for column, key in enumerate(node_keys):
                axis = axes[row, column]
                if mode == "amplitude":
                    last_image = axis.imshow(
                        np.clip(field_views[key][row] / amplitude_scales[key], 0.0, 1.0),
                        cmap="magma", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest", aspect="equal",
                    )
                    if column == 4:
                        _add_ols_inset(axis, trace["z_ols"], row, mode, z_scale)
                else:
                    last_image = axis.imshow(
                        field_views[key][row],
                        cmap="twilight", vmin=0.0, vmax=PI2, origin="lower", interpolation="nearest", aspect="equal",
                    )
                    if column == 4:
                        _add_ols_inset(axis, trace["z_ols"], row, mode, z_scale)
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(node_labels[column], fontsize=8, pad=5)
                if column == 0:
                    axis.set_ylabel(f"数字 {int(labels[row])} · #{row % DISPLAY_PER_CLASS + 1}", fontsize=7, rotation=0, ha="right", va="center", labelpad=15)
                for spine in axis.spines.values():
                    spine.set_visible(False)
        _require(last_image is not None, "endpoint trace figure is empty")
        if mode == "amplitude":
            scalar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="magma")
            scalar.set_array([])
            cbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), fraction=0.008, pad=0.006)
            cbar.set_label(f"振幅 / 节点内共同 q99.5（{scale_note}）", fontsize=7)
            title = f"{endpoint_label}：{len(node_keys)} 节点复场振幅"
        else:
            scalar = ScalarMappable(norm=Normalize(0.0, PI2), cmap="twilight")
            scalar.set_array([])
            cbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), fraction=0.008, pad=0.006, ticks=[0, math.pi / 2, math.pi, 3 * math.pi / 2, PI2])
            cbar.ax.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
            cbar.set_label("复场相位 arg(E)（rad），包裹到 [0, 2π)；低振幅像素留白", fontsize=7)
            title = f"{endpoint_label}：{len(node_keys)} 节点复场相位"
        cbar.ax.tick_params(labelsize=6.5)
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.966)
        fig.text(
            0.065,
            0.019,
            f"同一固定 DEV 选择：每类展示 2 个样本；不做逐图拉伸。相位在 |E| < 节点 q99.5 的 1% 处留白。OLS 小窗为实际 100 个复系数按行优先排列的 10×10 图。{endpoint_note}",
            fontsize=7,
            ha="left",
        )
        base = output_dir / f"{stem}_{mode}"
        main_axes = axes.ravel().tolist()
        _write_alignment_manifest(fig, main_axes, base)
        require_matplotlib_panel_alignment(
            fig,
            axes=main_axes,
            json_out=base.with_name(base.name + ".alignment.json"),
            overlay_svg=base.with_name(base.name + ".alignment.svg"),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True,
        )
        results.extend(_save_figure(fig, base))
    return results


def _run_endpoint_traces(root: Path, output_dir: Path, device: torch.device) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to overwrite existing endpoint visualization run: {output_dir}")
    required_paths = [
        root / CLOUD_CHECKPOINT,
        root / CLOUD_GENERATED_OLS,
        root / CLOUD_GENERATED_VISUALS,
        root / CLOUD_CONFIG,
        root / ENDPOINT_TRACE_SELECTION,
        *(root / item["checkpoint"] for item in M0_TRACE_ENDPOINTS),
    ]
    for path in required_paths:
        _require(path.is_file(), f"required endpoint visualization input is missing: {path}")

    authorities = robust._load_authorities(root)
    selected_images, selected_labels, source_indices, selection_manifest = _load_selection(root, authorities)
    selected_label_values = selected_labels.tolist()
    display_positions = [
        index
        for digit in range(10)
        for index in [position for position, label in enumerate(selected_label_values) if label == digit][:DISPLAY_PER_CLASS]
    ]
    class_seen = [sum(int(selected_labels[index]) == digit for index in display_positions) for digit in range(10)]
    _require(len(display_positions) == 10 * DISPLAY_PER_CLASS and class_seen == [DISPLAY_PER_CLASS] * 10, "fixed display selection no longer contains two examples per digit")
    display_images = selected_images[display_positions].contiguous()
    display_labels = selected_labels[display_positions].contiguous()
    display_source_indices = [source_indices[index] for index in display_positions]
    consumer, _models = _load_models(root, authorities, device)

    m0_traces: dict[str, dict[str, torch.Tensor]] = {}
    m0_metadata: list[dict[str, Any]] = []
    for endpoint in M0_TRACE_ENDPOINTS:
        trace, metadata = _capture_m0_endpoint(root, endpoint, consumer, display_images, device)
        m0_traces[str(endpoint["key"])] = trace
        m0_metadata.append(metadata)

    cloud_model, cloud_payload, cloud_metadata = _load_cloud_trace_model(root, device)
    cloud_ols_replay = _verify_cloud_generated_ols_replay(root, cloud_model, device)
    cloud_trace, cloud_trace_metadata = _capture_cloud_endpoint(cloud_model, display_images, device)

    m0_amplitude_scales: dict[str, float] = {}
    for key in NODE_KEYS:
        values = torch.cat([m0_traces[str(item["key"])][key].abs().flatten() for item in M0_TRACE_ENDPOINTS]).numpy()
        m0_amplitude_scales[key] = float(np.quantile(values, 0.995))
        _require(math.isfinite(m0_amplitude_scales[key]) and m0_amplitude_scales[key] > 0.0, f"invalid shared M0 amplitude scale: {key}")
    m0_z_values = torch.cat([m0_traces[str(item["key"])]["z_ols"].abs().flatten() for item in M0_TRACE_ENDPOINTS]).numpy()
    m0_z_scale = float(np.quantile(m0_z_values, 0.995))
    cloud_amplitude_scales = {
        key: float(np.quantile(cloud_trace[key].abs().flatten().numpy(), 0.995))
        for key in CLOUD_NODE_KEYS
    }
    for key, value in cloud_amplitude_scales.items():
        _require(math.isfinite(value) and value > 0.0, f"invalid cloud amplitude scale: {key}")
    cloud_z_scale = float(np.quantile(cloud_trace["z_ols"].abs().flatten().numpy(), 0.995))
    _require(math.isfinite(m0_z_scale) and m0_z_scale > 0.0 and math.isfinite(cloud_z_scale) and cloud_z_scale > 0.0, "invalid shared OLS amplitude scale")

    output_dir.mkdir(parents=True, exist_ok=False)
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    data_dir.mkdir()
    figures_dir.mkdir()
    selection_rows = [
        {"display_row": row + 1, "digit_label": int(display_labels[row]), "example_within_digit": row % DISPLAY_PER_CLASS + 1, "canonical_source_index": int(display_source_indices[row])}
        for row in range(len(display_labels))
    ]
    _write_csv(data_dir / "display_sample_manifest.csv", selection_rows)

    figure_files: list[str] = []
    m0_scale_note = "两组 M0 step1200 端点共用同一节点标尺"
    for endpoint in M0_TRACE_ENDPOINTS:
        key = str(endpoint["key"])
        figure_files.extend(
            _render_endpoint_trace(
                figures_dir,
                key,
                str(endpoint["label"]),
                m0_traces[key],
                NODE_KEYS,
                M0_NODE_LABELS_ZH,
                m0_amplitude_scales,
                m0_z_scale,
                display_labels,
                m0_scale_note,
                str(endpoint["status"]),
            )
        )
    figure_files.extend(
        _render_endpoint_trace(
            figures_dir,
            "cloud_large_step600",
            "云端大结构 step600（diagnostic-only）",
            cloud_trace,
            CLOUD_NODE_KEYS,
            CLOUD_NODE_LABELS,
            cloud_amplitude_scales,
            cloud_z_scale,
            display_labels,
            "云端架构内部各节点共用该节点标尺",
            "云端模型证据等级为 diagnostic-only；完整十节点来自真实 DEV 输入前向路径。",
        )
    )

    node_registry = {
        "m0": [
            {"index": index + 1, "key": key, "label": M0_NODE_LABELS_ZH[index], "shape": list(m0_traces[M0_TRACE_ENDPOINTS[0]["key"]][key].shape), "quantity": "complex field amplitude |E| and wrapped phase arg(E)"}
            for index, key in enumerate(NODE_KEYS)
        ],
        "cloud": [
            {"index": index + 1, "key": key, "label": CLOUD_NODE_LABELS[index], "plane_id": cloud_trace_metadata["plane_ids"][index], "shape": list(cloud_trace[key].shape), "quantity": "complex field amplitude |E| and wrapped phase arg(E)"}
            for index, key in enumerate(CLOUD_NODE_KEYS)
        ],
    }
    _write_json(data_dir / "node_registry.json", {"schema": "three_endpoint_native_node_registry.v1", **node_registry})
    _write_json(data_dir / "cloud_ols_replay_check.json", cloud_ols_replay)
    _write_json(data_dir / "endpoint_trace_metadata.json", {
        "schema": "three_endpoint_phase_amplitude_trace.v1",
        "status": "VALID_COMPLETED_DESCRIPTIVE_VISUALIZATION",
        "figure_claim": "Descriptive real-image optical-field trace of two M0 step1200 endpoints and the cloud step600 endpoint using the same fixed DEV examples; node count follows each architecture's native trace.",
        "inference_boundary": "No architecture-causal claim, hardware-readiness claim, or model acceptance is inferred. M0 pure BOGT parent remains INVALID_HARNESS; Cloud remains diagnostic-only.",
        "input_selection": {
            "selection_manifest": ENDPOINT_TRACE_SELECTION.as_posix(),
            "selection_manifest_sha256": file_sha256(root / ENDPOINT_TRACE_SELECTION),
            "selection_source_authority": "fixed real-DEV selection manifest and repository evaluator authority loaded by the pre-existing adapter",
            "selection_count": len(source_indices),
            "selection_per_class": 5,
            "display_rule": "first two examples per digit from the frozen 50-row DEV selection",
            "display_count": len(display_labels),
            "display_labels": display_labels.tolist(),
            "display_source_indices": display_source_indices,
            "all_three_endpoints_use_same_source_examples": True,
        },
        "endpoints": {
            "m0": m0_metadata,
            "cloud": {
                **cloud_metadata,
                **cloud_trace_metadata,
                "generated_ols_replay": cloud_ols_replay,
            },
        },
        "amplitude_display": {
            "normalization": "clip(|E| / q99.5, 0, 1); no per-image contrast stretch",
            "m0_shared_node_scales": m0_amplitude_scales,
            "m0_ols_q99_5": m0_z_scale,
            "cloud_node_scales": cloud_amplitude_scales,
            "cloud_ols_q99_5": cloud_z_scale,
            "cross_architecture_absolute_amplitude_comparison": "not implied; M0 and Cloud use separate node-wise scales because optical geometries and field conventions differ",
        },
        "phase_display": {
            "quantity": "actual angle of each captured complex field, wrapped to [0, 2π); no phase inferred from intensity",
            "mask": "pixels with |E| below 1% of that node's reported q99.5 amplitude scale are left blank because phase is unstable or undefined at zero amplitude",
        },
        "cmos_readout": "|cmos_field|^2 equals each model's raw CMOS intensity; Cloud square-law equality was checked at every trace batch",
        "path_separation": "The full node figures trace the same real DEV images through each model. The downloaded Cloud generated_ols.pt was separately replayed against its saved CMOS examples and was not spliced into the real-image encoder trace.",
        "training_optimizer_backward_or_checkpoint_write": False,
        "figures": figure_files,
        "code_sha256": file_sha256(Path(__file__)),
    })
    risk_contract = {
        "risk_level": "R2_ARTIFACT_PRODUCING_RESEARCH",
        "consequence": "A wrong node-to-plane mapping or amplitude scale could misstate how optical fields evolve across architectures.",
        "consumer": "Descriptive group-meeting and research figure; no training or promotion consumer.",
        "scope": "Three saved endpoints, fixed 20-row DEV display subset, native 8-node M0 and 10-node Cloud forward traces.",
        "invariants": ["same selected DEV source rows across endpoints", "strict checkpoint/config state load", "no optimizer/backward/update", "runtime/model state hashes unchanged", "CMOS intensity equals terminal complex-field modulus squared", "M0 and Cloud amplitude scales reported separately", "phase pixels below one percent of node q99.5 amplitude are masked"],
        "verification": ["checkpoint schemas and tensor geometries checked", "one fixed Cloud generated-OLS replay per stored first-five-per-class CMOS observation checked", "final figure alignment, PDF text, collision, and visual QA"],
        "stop_condition": "Stop before delivery on missing/malformed inputs, state mutation, node-count/shape mismatch, nonfinite fields, failed saved-OLS replay, or blocking figure QA.",
    }
    _write_json(output_dir / "visualization_contract.json", risk_contract)
    (output_dir / "README.md").write_text(
        "# 三端点原生节点复场可视化\n\n"
        "图中展示两个 M0 step1200 端点的八节点路径，以及云端 step600 大结构的十节点路径。所有端点使用同一固定 DEV 选择中的每类前两个样本，输入图像经过各自训练路径规定的本机分辨率与振幅编码。振幅采用节点内共同 q99.5 标尺，不做逐图拉伸；相位来自实际复场并包裹到 [0, 2π)，低于节点 q99.5 振幅 1% 的像素留白，以避免把零/极低振幅处不稳定的相位误读为有效信号。OLS 小窗显示同一前向样本的 100 个复系数。\n\n"
        "M0 纯 BOGT checkpoint 的父训练 attempt 仍为 INVALID_HARNESS，所以该面板只是描述性端点；M0 canonical 是有效 step1200 端点。云端模型保持 diagnostic-only。节点数量随架构变化，因此 M0 与 Cloud 面板不应被理解为逐列完全同构的架构因果比较。\n\n"
        "云端完整十节点图使用同一批真实 DEV 图像经过 SLM、编码器、光纤 OLS 和四层解码器的前向路径。下载的 generated_ols.pt 另行回放并与其保存的 CMOS 样本核对；它对应生成采样路径，没有与真实图像编码节点拼接。没有执行训练、优化器更新、反向传播或 checkpoint 写入。\n\n"
        "每个端点分别导出振幅与相位 PNG/PDF/SVG。数据来源、散列、固定样本索引、状态完整性、显示比例和风险/停止条件记录在 data/ 与 visualization_contract.json。\n",
        encoding="utf-8",
    )
    return {
        "result": "VALID_COMPLETED_DESCRIPTIVE_VISUALIZATION",
        "output_dir": output_dir.relative_to(root).as_posix(),
        "m0_nodes": len(NODE_KEYS),
        "cloud_nodes": len(CLOUD_NODE_KEYS),
        "display_samples": len(display_labels),
        "figure_count": len(figure_files),
        "cloud_generated_ols_replayed_rows": cloud_ols_replay["replayed_rows"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--endpoint-traces", action="store_true", help="render phase and amplitude at all native nodes for the two M0 and one Cloud endpoint")
    parser.add_argument("--scores", type=Path, default=None, help="completed blind readability score CSV; omit to replay candidates and write rating sheets")
    parser.add_argument("--reblind-existing", action="store_true", help="randomize aliases and columns for a pending run without replaying checkpoints")
    args = parser.parse_args()
    _figure_style()
    default_output = ENDPOINT_TRACE_OUTPUT_RELATIVE if args.endpoint_traces else OUTPUT_RELATIVE
    chosen_output = args.output_dir if args.output_dir is not None else default_output
    output_dir = chosen_output if chosen_output.is_absolute() else ROOT / chosen_output
    output_dir = output_dir.resolve()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("fixed DEV optical replay requires an available CUDA device")
    cuda_runtime = robust._prepare_cuda_runtime()
    if args.endpoint_traces:
        _require(args.scores is None and not args.reblind_existing, "--endpoint-traces cannot be combined with --scores or --reblind-existing")
        matplotlib.rcParams.update({"font.family": ["Microsoft YaHei", "Arial", "DejaVu Sans", "sans-serif"]})
        result = _run_endpoint_traces(ROOT, output_dir, torch.device("cuda:0"))
    elif args.reblind_existing:
        _require(args.scores is None, "--reblind-existing and --scores cannot be combined")
        result = _reblind_existing(output_dir)
    else:
        result = _prepare(ROOT, output_dir, cuda_runtime) if args.scores is None else _finalize(ROOT, output_dir, args.scores.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
