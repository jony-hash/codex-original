#!/usr/bin/env python3
"""Zero-training v63 code-conditioning provenance and crossed-factor audit.

The readable split was selected from teacher targets and therefore can make
otherwise IID one-shot codes appear label-associated.  This diagnostic verifies
the code-to-teacher-noise provenance before measuring a balanced 100-code by
10-label crossed design through the frozen v60 physical student.  It is not a
student stage, does not refit a software decoder, and cannot promote a raw
CMOS image that remains unreadable.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project import OneShotOpticalGenerator, load_one_shot_checkpoint, load_training_config
from ldm_project.generation.codes import _sample_one_shot_codes_with_seed
from ldm_project.teacher.distill import _project_codes_to_teacher_noise


REPO_ROOT = Path(__file__).resolve().parents[1]
V62_TOOL = REPO_ROOT / "tools/diagnose_v62_native_conditional_control_audit.py"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs/20260715_v60/runs/"
    "20260715_010531_532021_train_v60_coherent_relay_full_fov_from_scratch/"
    "checkpoints/one_shot_distill_best.pt"
)
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs/train_v56_coherent_relay_from_scratch.yaml"
DEFAULT_REFERENCE_REPORT = REPO_ROOT / "outputs/reports/v58_3_b0_relay_cuda_preflight.json"
DEFAULT_V61_REPORT = REPO_ROOT / "outputs/reports/v61_offline_decodability_upper_bound.json"
DEFAULT_V61_TENSORS = REPO_ROOT / "outputs/reports/v61_offline_decodability_upper_bound.pt"
DEFAULT_V62_REPORT = REPO_ROOT / "outputs/reports/v62_native_conditional_control_audit.json"
DEFAULT_V62_TENSORS = REPO_ROOT / "outputs/reports/v62_native_conditional_control_audit.pt"
DEFAULT_DATASET_ROOT = (
    REPO_ROOT
    / "outputs/20260514_unversioned/runs/"
    "20260610_151747_768931_train_teacher_distill_15k_acceptance_matched_"
    "diffusion_e40_gs40_readable_split_from_full15k_cuda/artifacts"
)
DEFAULT_DATASETS = {
    "train": DEFAULT_DATASET_ROOT / "teacher_distill_dataset_readable_train.pt",
    "val": DEFAULT_DATASET_ROOT / "teacher_distill_dataset_readable_val.pt",
    "test": DEFAULT_DATASET_ROOT / "teacher_distill_dataset_readable_test.pt",
}
DEFAULT_OUTPUT = REPO_ROOT / "outputs/reports/v63_code_conditioning_provenance_audit.json"

PLANE_ORDER = ("slm_command", "pre_encoder_intensity", "chip_intensity", "camera_intensity")
SOURCE_SAMPLE_COUNT = 15_000
SOURCE_SAMPLES_PER_CLASS = 1_500
FULL_EXPORT_PERMUTATIONS = 16
SPLIT_PERMUTATIONS = 128
GENERATOR_CODE_FRACTION_MIN = 0.25
CODE_RETENTION_FRACTION = 0.10
V62_RESIDUAL_CORR_MIN = 0.10


def _load_v62_module():
    spec = importlib.util.spec_from_file_location("v62_conditional_control", V62_TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v62 helper tool: {V62_TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V62 = _load_v62_module()


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _invalid_report(*, inputs: Mapping[str, Path], reasons: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "v63-code-conditioning-provenance-audit",
        "status": "harness-invalid",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "inputs": {name: str(path) for name, path in inputs.items()},
        "reasons": reasons,
    }


def _load_bundle(path: Path) -> dict[str, Any]:
    return V62._load_bundle(path)


def _permutation_null(
    values: torch.Tensor,
    labels: torch.Tensor,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float | int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    observed = V62.prototype_cv_accuracy(values, labels)
    null = torch.tensor(
        [
            V62.prototype_cv_accuracy(
                values, labels[torch.randperm(labels.numel(), generator=generator)]
            )
            for _ in range(permutations)
        ]
    )
    return {
        "observed_accuracy": observed,
        "permutation_count": permutations,
        "seed": seed,
        "null_mean": float(null.mean()),
        "null_p95": float(torch.quantile(null, 0.95)),
        "p_value": float(((null >= observed).sum() + 1) / (permutations + 1)),
    }


def _source_labels() -> torch.Tensor:
    return torch.arange(10, dtype=torch.long).repeat_interleave(SOURCE_SAMPLES_PER_CLASS)


def _validated_provenance(
    datasets: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any], torch.Tensor]:
    """Replay the code/noise chain without loading the multi-gigabyte source payload."""
    reasons: list[str] = []
    source_codes = _sample_one_shot_codes_with_seed(
        torch, SOURCE_SAMPLE_COUNT, 128, 1234, "cpu"
    ).float()
    source_labels = _source_labels()
    split_reports: dict[str, Any] = {}
    for index, (split, payload) in enumerate(datasets.items()):
        config = payload.get("config")
        if not isinstance(config, Mapping):
            reasons.append(f"{split}: missing distillation config")
            continue
        if (
            config.get("samples_per_class") != SOURCE_SAMPLES_PER_CLASS
            or config.get("code_seed") != 1234
            or config.get("code_projection_seed") != 2026
            or list(config.get("class_labels", ())) != list(range(10))
        ):
            reasons.append(f"{split}: source code provenance metadata is not the pre-registered export")
            continue
        indices = payload.get("readability_filter_indices")
        codes = payload.get("codes")
        labels = payload.get("class_labels")
        noise = payload.get("teacher_initial_noise")
        if not torch.is_tensor(indices) or tuple(indices.shape) != (V62.SPLIT_COUNTS[split],):
            reasons.append(f"{split}: readability_filter_indices has an invalid shape")
            continue
        if not torch.is_tensor(codes) or not torch.is_tensor(labels) or not torch.is_tensor(noise):
            reasons.append(f"{split}: codes, labels, or teacher_initial_noise is missing")
            continue
        indices = indices.detach().cpu().long()
        if int(indices.min()) < 0 or int(indices.max()) >= SOURCE_SAMPLE_COUNT or int(indices.unique().numel()) != int(indices.numel()):
            reasons.append(f"{split}: filter indices are not unique source indices")
            continue
        expected_codes = source_codes.index_select(0, indices)
        expected_labels = source_labels.index_select(0, indices)
        replayed_noise = _project_codes_to_teacher_noise(
            torch,
            codes.detach().cpu().float(),
            tuple(noise.shape),
            2026,
            "cpu",
        ).float()
        if not torch.equal(codes.detach().cpu().float(), expected_codes):
            reasons.append(f"{split}: saved codes do not match the seeded source export")
        if not torch.equal(labels.detach().cpu().long(), expected_labels):
            reasons.append(f"{split}: saved labels do not match source indices")
        if not torch.equal(noise.detach().cpu().float(), replayed_noise):
            reasons.append(f"{split}: teacher_initial_noise does not exactly replay from codes")
        split_reports[split] = {
            "source_index_min": int(indices.min()),
            "source_index_max": int(indices.max()),
            "source_index_unique_count": int(indices.unique().numel()),
            "code_replay_exact": bool(torch.equal(codes.detach().cpu().float(), expected_codes)),
            "label_replay_exact": bool(torch.equal(labels.detach().cpu().long(), expected_labels)),
            "teacher_initial_noise_replay_exact": bool(torch.equal(noise.detach().cpu().float(), replayed_noise)),
            "code_label_prototype": _permutation_null(
                codes.detach().cpu().float()[:, None, None, :],
                labels.detach().cpu().long(),
                permutations=SPLIT_PERMUTATIONS,
                seed=6301 + index,
            ),
        }
    full_report = {
        "sample_count": SOURCE_SAMPLE_COUNT,
        "generation": "one global seeded IID Gaussian code array before labels are assigned",
        "code_label_prototype": _permutation_null(
            source_codes[:, None, None, :],
            source_labels,
            permutations=FULL_EXPORT_PERMUTATIONS,
            seed=6300,
        ),
    }
    selection_confound = any(
        float(item["code_label_prototype"]["p_value"]) < 0.01
        for item in split_reports.values()
    )
    return reasons, {
        "full_export": full_report,
        "readable_splits": split_reports,
        "selection_confound": selection_confound,
        "interpretation": (
            "readable-split code-label association is a target-selection property, not a code-conditioning score"
            if selection_confound
            else "no significant readable-split code-label association at the pre-registered null"
        ),
    }, source_codes


def _validate_v62_artifacts(
    report_path: Path,
    tensor_path: Path,
    *,
    checkpoint: Path,
    test_payload: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    pairing: dict[str, Any] = {}
    if not report_path.is_file() or not tensor_path.is_file():
        return ["v62 report or tensor artifact is missing"], pairing
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "v62-native-conditional-control-audit":
        reasons.append("v62 report schema is invalid")
    if report.get("diagnostic_only") is not True or report.get("accepted_summary_written") is not False:
        reasons.append("v62 report diagnostic metadata is invalid")
    report_checkpoint = Path(str(report.get("inputs", {}).get("checkpoint", "")))
    if report_checkpoint != checkpoint:
        reasons.append("v62 report checkpoint does not match v60")
    pairing = report.get("pairing", {})
    if not isinstance(pairing, Mapping):
        reasons.append("v62 report lacks pairing metrics")
        pairing = {}
    bundle = _load_bundle(tensor_path)
    expected_labels = test_payload.get("class_labels")
    expected_target = test_payload.get("teacher_camera")
    if bundle.get("schema_version") != "v62-native-conditional-control-audit":
        reasons.append("v62 tensor schema is invalid")
    if not torch.is_tensor(bundle.get("class_labels")) or not torch.equal(bundle["class_labels"].long(), expected_labels.long()):
        reasons.append("v62 tensor labels do not match readable test data")
    if not torch.is_tensor(bundle.get("teacher_target")) or not torch.equal(bundle["teacher_target"].float(), expected_target.float()):
        reasons.append("v62 tensor targets do not match readable test data")
    return reasons, dict(pairing)


def crossed_codes_and_labels(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(codes.shape) != (100, 128):
        raise ValueError(f"crossed design requires test codes [100,128], got {tuple(codes.shape)}")
    labels = torch.arange(10, dtype=torch.long)
    return (
        codes.detach().cpu().float().unsqueeze(1).expand(-1, 10, -1).reshape(-1, 128).clone(),
        labels.unsqueeze(0).expand(codes.shape[0], -1).reshape(-1).clone(),
    )


def _batch_planes(outputs: Mapping[str, Any], *, config) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    camera, errors = V62._validate_forward(outputs, config=config)
    fields = outputs["optical_fields"]
    front = V62._field_from_record(fields, "pre_encoder_relay_image")
    chip = V62._field_from_record(fields, "chip")
    return {
        "slm_command": outputs["slm_command"].detach().cpu().float(),
        "pre_encoder_intensity": front.abs().square().detach().cpu().float(),
        "chip_intensity": chip.abs().square().detach().cpu().float(),
        "camera_intensity": camera,
    }, errors


def _forward_crossed(
    model: torch.nn.Module,
    *,
    codes: torch.Tensor,
    labels: torch.Tensor,
    original_labels: torch.Tensor,
    preview_code_indices: list[int],
    config,
    device: torch.device,
    batch_size: int,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, torch.Tensor]],
    dict[str, torch.Tensor],
    torch.Tensor,
    dict[str, float],
]:
    accumulators: dict[str, _NativeFactorAccumulator] = {}
    errors = {
        "raw_camera_intensity_field_max_error": 0.0,
        "detector_normalization_max_error": 0.0,
    }
    with torch.no_grad():
        for start in range(0, int(codes.shape[0]), batch_size):
            end = min(start + batch_size, int(codes.shape[0]))
            outputs = model(codes[start:end].to(device), labels[start:end].to(device), return_optical_fields=True)
            planes, batch_errors = _batch_planes(outputs, config=config)
            code_indices = torch.arange(start, end, dtype=torch.long).div(10, rounding_mode="floor")
            for name in PLANE_ORDER:
                if name not in accumulators:
                    accumulators[name] = _NativeFactorAccumulator(
                        planes[name],
                        original_labels=original_labels,
                        preview_code_indices=preview_code_indices,
                    )
                accumulators[name].update(planes[name], code_indices, labels[start:end].cpu())
            for name, value in batch_errors.items():
                errors[name] = max(errors[name], value)
            del outputs, planes
            torch.cuda.empty_cache()
    finalized = {name: accumulator.finalize() for name, accumulator in accumulators.items()}
    factors = {name: result[0] for name, result in finalized.items()}
    means = {name: result[1] for name, result in finalized.items()}
    diagonal = {name: result[2] for name, result in finalized.items()}
    return factors, means, diagonal, finalized["camera_intensity"][3], errors


def balanced_factor_decomposition(values: torch.Tensor) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Exact balanced code x label ANOVA in the native plane coordinate system."""
    if values.ndim != 5 or tuple(values.shape[:2]) != (100, 10):
        raise ValueError(f"factor values must have shape [100,10,C,H,W], got {tuple(values.shape)}")
    global_mean = values.mean(dim=(0, 1), keepdim=True)
    code_mean = values.mean(dim=1, keepdim=True)
    label_mean = values.mean(dim=0, keepdim=True)
    code_energy = (code_mean - global_mean).square().mean()
    label_energy = (label_mean - global_mean).square().mean()
    interaction = values - code_mean - label_mean + global_mean
    interaction_energy = interaction.square().mean()
    total_energy = (values - global_mean).square().mean().clamp_min(1.0e-20)
    return {
        "total_centered_energy": float(total_energy),
        "code_main_energy": float(code_energy),
        "label_main_energy": float(label_energy),
        "interaction_energy": float(interaction_energy),
        "code_fraction": float(code_energy / total_energy),
        "label_fraction": float(label_energy / total_energy),
        "interaction_fraction": float(interaction_energy / total_energy),
        "fraction_sum": float((code_energy + label_energy + interaction_energy) / total_energy),
    }, {
        "global_mean": global_mean.squeeze(0).squeeze(0).detach().cpu(),
        "code_mean": code_mean.squeeze(1).detach().cpu(),
        "label_mean": label_mean.squeeze(0).detach().cpu(),
    }


class _NativeFactorAccumulator:
    """Stream a balanced native plane without retaining its 1000 crossed fields."""

    def __init__(
        self,
        example: torch.Tensor,
        *,
        original_labels: torch.Tensor,
        preview_code_indices: list[int],
    ) -> None:
        if example.ndim != 4:
            raise ValueError(f"native plane batch must be [N,C,H,W], got {tuple(example.shape)}")
        shape = tuple(example.shape[1:])
        self.pixel_count = int(example[0].numel())
        self.total_sum = torch.zeros(shape, dtype=torch.float32)
        self.code_sum = torch.zeros((100, *shape), dtype=torch.float32)
        self.label_sum = torch.zeros((10, *shape), dtype=torch.float32)
        self.total_square_sum = 0.0
        self.diagonal = torch.empty((100, *shape), dtype=torch.float32)
        self.preview = torch.empty((10, 10, *shape), dtype=torch.float32)
        self.original_labels = original_labels.detach().cpu().long()
        self.preview_rows = {code_index: row for row, code_index in enumerate(preview_code_indices)}

    def update(self, values: torch.Tensor, code_indices: torch.Tensor, labels: torch.Tensor) -> None:
        values = values.detach().cpu().float()
        code_indices, labels = code_indices.detach().cpu().long(), labels.detach().cpu().long()
        self.total_sum.add_(values.sum(dim=0))
        self.total_square_sum += float(values.square().sum())
        self.code_sum.index_add_(0, code_indices, values)
        self.label_sum.index_add_(0, labels, values)
        diagonal = labels == self.original_labels.index_select(0, code_indices)
        if bool(diagonal.any()):
            self.diagonal.index_copy_(0, code_indices[diagonal], values[diagonal])
        for local_index, code_index in enumerate(code_indices.tolist()):
            row = self.preview_rows.get(code_index)
            if row is not None:
                self.preview[row, int(labels[local_index])] = values[local_index]

    def finalize(self) -> tuple[dict[str, float], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        global_mean = self.total_sum / 1_000.0
        code_mean = self.code_sum / 10.0
        label_mean = self.label_sum / 100.0
        mean_square = self.total_square_sum / float(1_000 * self.pixel_count)
        global_square = float(global_mean.square().mean())
        total_energy = max(mean_square - global_square, 1.0e-20)
        code_energy = max(float(code_mean.square().mean()) - global_square, 0.0)
        label_energy = max(float(label_mean.square().mean()) - global_square, 0.0)
        interaction_energy = max(total_energy - code_energy - label_energy, 0.0)
        factors = {
            "total_centered_energy": total_energy,
            "code_main_energy": code_energy,
            "label_main_energy": label_energy,
            "interaction_energy": interaction_energy,
            "code_fraction": code_energy / total_energy,
            "label_fraction": label_energy / total_energy,
            "interaction_fraction": interaction_energy / total_energy,
            "fraction_sum": (code_energy + label_energy + interaction_energy) / total_energy,
        }
        means = {
            "global_mean": global_mean,
            "code_mean": code_mean,
            "label_mean": label_mean,
        }
        return factors, means, self.diagonal, self.preview


def _v62_residual_is_valid(pairing: Mapping[str, Any]) -> bool:
    try:
        return (
            float(pairing["matched_residual"]["corr_mean"]) >= V62_RESIDUAL_CORR_MIN
            and float(pairing["residual_corr_vs_within_class_target_rotation"]["ci_low"]) > 0.0
        )
    except (KeyError, TypeError, ValueError):
        return False


def _decision(factors: Mapping[str, Mapping[str, float]], *, v62_residual_valid: bool) -> dict[str, Any]:
    slm = factors["slm_command"]
    generator_code_dominant = (
        float(slm["code_fraction"]) >= GENERATOR_CODE_FRACTION_MIN
        and float(slm["code_fraction"]) > float(slm["label_fraction"])
    )
    localized_plane = None
    for index, name in enumerate(PLANE_ORDER[1:], start=1):
        candidate = factors[name]
        code_lost = float(candidate["code_fraction"]) <= CODE_RETENTION_FRACTION * float(slm["code_fraction"])
        label_retained = float(candidate["label_fraction"]) >= CODE_RETENTION_FRACTION * float(slm["label_fraction"])
        downstream_lost = all(
            float(factors[later]["code_fraction"]) <= CODE_RETENTION_FRACTION * float(slm["code_fraction"])
            for later in PLANE_ORDER[index:]
        )
        if code_lost and label_retained and downstream_lost:
            localized_plane = name
            break
    camera_retained = float(factors["camera_intensity"]["code_fraction"]) > CODE_RETENTION_FRACTION * float(slm["code_fraction"])
    if localized_plane is not None:
        return {
            "status": "localized-code-factor-attenuation-candidate",
            "first_plane": localized_plane,
            "generator_code_dominant": generator_code_dominant,
            "camera_code_retained": camera_retained,
            "next_gate": "isolate this plane with a zero-training native code-factor audit; do not train",
        }
    if generator_code_dominant and camera_retained and v62_residual_valid:
        return {
            "status": "native-code-conditioned-subvisual-representation-confirmed",
            "generator_code_dominant": True,
            "camera_code_retained": True,
            "next_gate": "isolate representation/readability without decoder or loss tuning; do not promote",
        }
    if generator_code_dominant:
        return {
            "status": "generator-code-dominant-without-confirmed-camera-retention",
            "generator_code_dominant": True,
            "camera_code_retained": camera_retained,
            "next_gate": "audit code-factor retention across formal planes; do not train",
        }
    return {
        "status": "generator-code-factor-not-dominant",
        "generator_code_dominant": False,
        "camera_code_retained": camera_retained,
        "next_gate": "audit generator conditioning source before any architecture or loss change; do not train",
    }


def _render_preview(output: Path, *, crossed_camera: torch.Tensor) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render the v63 crossed preview") from exc
    figure, axes = plt.subplots(10, 10, figsize=(15, 15), squeeze=False)
    for row in range(10):
        for label in range(10):
            image = crossed_camera[row, label, 0].detach().cpu().float()
            scale = image.quantile(0.995).clamp_min(1.0e-12)
            axes[row, label].imshow((image / scale).clamp(0.0, 1.0), cmap="gray", interpolation="nearest")
            axes[row, label].axis("off")
            if row == 0:
                axes[row, label].set_title(f"label {label}")
        axes[row, 0].set_ylabel(f"code from class {row}")
    figure.suptitle("v63 crossed code x label physical CMOS — diagnostic only", y=0.999)
    figure.tight_layout()
    preview = output.with_name(f"{output.stem}_preview.png")
    figure.savefig(preview, dpi=160)
    plt.close(figure)
    return preview


def run(
    *,
    train_config: str | Path = DEFAULT_TRAIN_CONFIG,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    reference_report: str | Path = DEFAULT_REFERENCE_REPORT,
    v61_report: str | Path = DEFAULT_V61_REPORT,
    v61_tensors: str | Path = DEFAULT_V61_TENSORS,
    v62_report: str | Path = DEFAULT_V62_REPORT,
    v62_tensors: str | Path = DEFAULT_V62_TENSORS,
    train_dataset: str | Path = DEFAULT_DATASETS["train"],
    val_dataset: str | Path = DEFAULT_DATASETS["val"],
    test_dataset: str | Path = DEFAULT_DATASETS["test"],
    output: str | Path = DEFAULT_OUTPUT,
    device: str = "cuda",
    batch_size: int = 4,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    paths = {
        "train_config": Path(train_config),
        "checkpoint": Path(checkpoint),
        "reference_report": Path(reference_report),
        "v61_report": Path(v61_report),
        "v61_tensors": Path(v61_tensors),
        "v62_report": Path(v62_report),
        "v62_tensors": Path(v62_tensors),
        "train_dataset": Path(train_dataset),
        "val_dataset": Path(val_dataset),
        "test_dataset": Path(test_dataset),
    }
    output_path = Path(output)
    reasons = [f"missing input: {name}={path}" for name, path in paths.items() if not path.is_file()]
    if paths["checkpoint"].resolve() != DEFAULT_CHECKPOINT.resolve():
        reasons.append("v63 accepts only the pre-registered v60 epoch-5 checkpoint")
    if device != "cuda":
        reasons.append("v63 crossed full-geometry diagnostic requires --device cuda")
    if not torch.cuda.is_available():
        reasons.append("CUDA is unavailable")
    if reasons:
        report = _invalid_report(inputs=paths, reasons=reasons)
        _json_dump(output_path, report)
        return report
    config = load_training_config(paths["train_config"])
    reasons.extend(V62._validate_contract(config))
    reasons.extend(V62._validate_v58_3_report(paths["reference_report"]))
    datasets = {split: _load_bundle(paths[f"{split}_dataset"]) for split in V62.SPLIT_COUNTS}
    for split, payload in datasets.items():
        reasons.extend(V62._validate_readable_split(payload, split))
    v61_reasons, _ = V62._validate_upper_bound(
        paths["v61_report"],
        paths["v61_tensors"],
        checkpoint=paths["checkpoint"],
        test_payload=datasets["test"],
    )
    reasons.extend(v61_reasons)
    v62_reasons, v62_pairing = _validate_v62_artifacts(
        paths["v62_report"],
        paths["v62_tensors"],
        checkpoint=paths["checkpoint"],
        test_payload=datasets["test"],
    )
    reasons.extend(v62_reasons)
    provenance_reasons, provenance, _ = _validated_provenance(datasets)
    reasons.extend(provenance_reasons)
    if reasons:
        report = _invalid_report(inputs=paths, reasons=reasons)
        _json_dump(output_path, report)
        return report
    test_codes = datasets["test"]["codes"].detach().cpu().float()
    test_labels = datasets["test"]["class_labels"].detach().cpu().long()
    crossed_codes, crossed_labels = crossed_codes_and_labels(test_codes)
    resolved_device = torch.device("cuda")
    model = OneShotOpticalGenerator(config.physics, config.one_shot, image_size=config.first_stage.image_size).to(resolved_device)
    load_one_shot_checkpoint(model, torch.load(paths["checkpoint"], map_location=resolved_device, weights_only=False))
    model.eval()
    before = V62._state_snapshot(model)
    report: dict[str, Any] = {
        "schema_version": "v63-code-conditioning-provenance-audit",
        "status": "manual-review-required",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "inputs": {name: str(path) for name, path in paths.items()},
        "invariants": {
            "student_training": False,
            "optimizer_created": False,
            "parameter_update_count": 0,
            "formal_contract_changed": False,
            "target_code_or_bridge": "disabled",
            "crossed_design": "100 held-out test codes x all 10 labels",
            "software_readout_refit": False,
            "native_sampling_only": True,
        },
        "pre_registered": {
            "generator_code_fraction_min": GENERATOR_CODE_FRACTION_MIN,
            "downstream_code_retention_fraction": CODE_RETENTION_FRACTION,
            "v62_residual_corr_min": V62_RESIDUAL_CORR_MIN,
            "full_export_permutations": FULL_EXPORT_PERMUTATIONS,
            "split_permutations": SPLIT_PERMUTATIONS,
        },
        "provenance": provenance,
        "progress": [],
        "reasons": [],
    }
    _json_dump(output_path, report)
    try:
        preview_code_indices = [int(torch.nonzero(test_labels == label, as_tuple=False)[0]) for label in range(10)]
        factors, means, diagonal, camera_preview, semantic_errors = _forward_crossed(
            model,
            codes=crossed_codes,
            labels=crossed_labels,
            original_labels=test_labels,
            preview_code_indices=preview_code_indices,
            config=config,
            device=resolved_device,
            batch_size=batch_size,
        )
        report["progress"].append("crossed_native_forward")
        _json_dump(output_path, report)
        changed = V62._state_changes(model, before)
        report["invariants"]["parameter_update_count"] = len(changed)
        if changed:
            report["status"] = "harness-invalid"
            report["reasons"].append(f"model parameters changed during diagnostic: {changed[:5]}")
            return report
        report["field_semantic_checks"] = semantic_errors
        report["factors"] = factors
        report["v62_paired_residual_reference"] = {
            "matched_residual": v62_pairing.get("matched_residual"),
            "residual_corr_vs_within_class_target_rotation": v62_pairing.get("residual_corr_vs_within_class_target_rotation"),
            "valid": _v62_residual_is_valid(v62_pairing),
            "interpretation": "reference only; v63 does not refit or use a software readout",
        }
        report["decision"] = _decision(
            factors,
            v62_residual_valid=_v62_residual_is_valid(v62_pairing),
        )
        preview = _render_preview(output_path, crossed_camera=camera_preview)
        tensor_path = output_path.with_suffix(".pt")
        torch.save(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "accepted_summary_written": False,
                "test_codes": test_codes,
                "test_class_labels": test_labels,
                "diagonal_native_planes": diagonal,
                "factor_means": means,
            },
            tensor_path,
        )
        report["artifacts"] = {"tensors": str(tensor_path), "preview": str(preview)}
        report.pop("progress", None)
    except (RuntimeError, ValueError) as exc:
        report["status"] = "harness-invalid"
        report["reasons"].append(str(exc))
    finally:
        _json_dump(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference-report", type=Path, default=DEFAULT_REFERENCE_REPORT)
    parser.add_argument("--v61-report", type=Path, default=DEFAULT_V61_REPORT)
    parser.add_argument("--v61-tensors", type=Path, default=DEFAULT_V61_TENSORS)
    parser.add_argument("--v62-report", type=Path, default=DEFAULT_V62_REPORT)
    parser.add_argument("--v62-tensors", type=Path, default=DEFAULT_V62_TENSORS)
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_DATASETS["train"])
    parser.add_argument("--val-dataset", type=Path, default=DEFAULT_DATASETS["val"])
    parser.add_argument("--test-dataset", type=Path, default=DEFAULT_DATASETS["test"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({"status": result["status"], "decision": result.get("decision"), "reasons": result.get("reasons", [])}, sort_keys=True))
    raise SystemExit(0 if result["status"] != "harness-invalid" else 1)


if __name__ == "__main__":
    main()
