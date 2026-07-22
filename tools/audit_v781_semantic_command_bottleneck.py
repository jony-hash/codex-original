"""Generator-only v78.1 semantic-command bottleneck postmortem."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ldm_project import load_training_config
from ldm_project.student.checkpoints import checkpoint_boundary_contract, load_one_shot_checkpoint
from ldm_project.student.experimental.model import OneShotOpticalGenerator
from ldm_project.student.semantic_command import project_residual_to_basis, semantic_basis
from tools.audit_v73_semantic_support import tensor_hash
from tools.diagnose_v68_transparent_memory import _build_prototype_bank

AMPLITUDE = 0.4027685085311532
BACKGROUND_FLOOR = 0.05
CODE_COUNT = 100
LABEL_COUNT = 10


class _StreamingStage:
    def __init__(self, code_count: int, label_count: int) -> None:
        self.code_sum: torch.Tensor | None = None
        self.label_sum: torch.Tensor | None = None
        self.total_sum: torch.Tensor | None = None
        self.total_sq: torch.Tensor | None = None
        self.count = 0
        self.code_count = code_count
        self.label_count = label_count
        self.digest = hashlib.sha256()

    def add(self, code_start: int, values: torch.Tensor) -> None:
        if values.ndim < 3 or values.shape[0] != values.shape[1]:
            raise ValueError("stage values must be [codes, labels, ...]")
        values = values.detach().float().cpu()
        n_codes, n_labels = values.shape[:2]
        flat = values.reshape(n_codes, n_labels, -1).double()
        code_sum = flat.sum(1)
        label_sum = flat.sum(0)
        total_sum = flat.sum((0, 1))
        total_sq = flat.square().sum((0, 1))
        if self.code_sum is None:
            self.code_sum = torch.zeros(self.code_count, flat.shape[-1], dtype=torch.float64)
            self.label_sum = torch.zeros(self.label_count, flat.shape[-1], dtype=torch.float64)
            self.total_sum = torch.zeros(flat.shape[-1], dtype=torch.float64)
            self.total_sq = torch.zeros(flat.shape[-1], dtype=torch.float64)
        self.code_sum[code_start : code_start + n_codes] += code_sum
        if self.label_sum is None:
            raise ValueError("stage label statistics are not initialized")
        self.label_sum += label_sum
        self.total_sum += total_sum
        self.total_sq += total_sq
        self.count += n_codes * n_labels
        self.digest.update(values.contiguous().numpy().tobytes())

    def report(self) -> dict[str, Any]:
        if self.code_sum is None or self.label_sum is None or self.total_sum is None or self.total_sq is None:
            raise ValueError("stage has no samples")
        grand = self.total_sum / float(self.count)
        code_mean = self.code_sum / float(self.label_count)
        label_mean = self.label_sum / float(self.code_count)
        total_ss = float((self.total_sq - self.count * grand.square()).sum())
        code_ss = float(self.label_count * (code_mean - grand).square().sum())
        label_ss = float(self.code_count * (label_mean - grand).square().sum())
        interaction_ss = max(total_ss - code_ss - label_ss, 0.0)
        total_ss = max(total_ss, 1.0e-12)
        return {
            "code_effect_ss": code_ss,
            "label_effect_ss": label_ss,
            "interaction_effect_ss": interaction_ss,
            "total_ss": total_ss,
            "code_fraction": code_ss / total_ss,
            "label_fraction": label_ss / total_ss,
            "interaction_fraction": interaction_ss / total_ss,
            "finite": bool(torch.isfinite(self.code_sum).all() and torch.isfinite(self.label_sum).all() and torch.isfinite(self.total_sum).all()),
            "value_hash": self.digest.hexdigest(),
            "sample_count": self.count,
        }


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(*, config: str | Path, checkpoint: str | Path, validation_dataset: str | Path, v78_admission: str | Path, v78_decision: str | Path, output: str | Path) -> dict[str, Any]:
    cfg = load_training_config(config)
    admission = _read(v78_admission)
    decision = _read(v78_decision)
    if admission.get("status") != "v78-cpu-admission-failed" or decision.get("status") != "v78-command-code-factor-insufficient":
        raise ValueError("v78.1 requires the frozen valid v78 code-factor failure")
    data = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes = data["codes"].float()
    labels = data["class_labels"].long()
    if tuple(codes.shape) != (CODE_COUNT, 128) or set(labels.tolist()) != set(range(LABEL_COUNT)):
        raise ValueError("v78.1 requires the fixed validation split")
    prototypes = _build_prototype_bank(cfg).float().contiguous()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = OneShotOpticalGenerator(cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size).eval()
    load_one_shot_checkpoint(model, payload)
    before_state_hash = _state_hash(model)
    before_keys = tuple(sorted(model.state_dict()))
    basis = semantic_basis(200)
    stages = {name: _StreamingStage(CODE_COUNT, LABEL_COUNT) for name in ("residual_basis_coefficients", "unit_coefficient_direction", "projected_pattern", "weighted_zero_mean_pattern", "template_support_pattern", "normalized_pattern", "command_delta", "command", "physical_slm_intensity")}
    support_energy = torch.zeros((), dtype=torch.float64)
    delta_energy = torch.zeros((), dtype=torch.float64)
    reconstruction_errors = {name: 0.0 for name in ("basis", "weighted_mean", "normalization", "command")}
    with torch.no_grad():
        for code_start in range(0, CODE_COUNT, 10):
            code_batch = codes[code_start : code_start + 10]
            code_input = code_batch.repeat_interleave(LABEL_COUNT, dim=0)
            label_input = torch.arange(LABEL_COUNT).repeat(code_batch.shape[0])
            command, diagnostics = model.generator_e.forward_with_diagnostics(code_input, label_input)
            template = model.generator_e.template_logits.index_select(0, label_input)
            residual = diagnostics["generator_code_delta_logits"]
            projected, coefficients = project_residual_to_basis(residual, basis)
            support = torch.sigmoid(template)
            weighted_mean = (support * projected).flatten(1).sum(1, keepdim=True) / support.flatten(1).sum(1, keepdim=True).clamp_min(1.0e-8)
            weighted_zero_mean = projected - weighted_mean[:, :, None, None]
            support_pattern = support * weighted_zero_mean
            max_abs = support_pattern.flatten(1).abs().amax(1, keepdim=True).clamp_min(1.0e-8)
            normalized = support_pattern / max_abs[:, :, None, None]
            command_delta = AMPLITUDE * normalized
            base_command = ((support.sqrt() - BACKGROUND_FLOOR) / (1.0 - BACKGROUND_FLOOR)).clamp(0.0, 1.0)
            rebuilt_command = (base_command + command_delta).clamp(0.0, 1.0)
            intensity = (BACKGROUND_FLOOR + (1.0 - BACKGROUND_FLOOR) * rebuilt_command).square()
            batch_values = {
                "residual_basis_coefficients": coefficients.reshape(10, 10, 8),
                "unit_coefficient_direction": (coefficients / coefficients.norm(dim=1, keepdim=True).clamp_min(1.0e-8)).reshape(10, 10, 8),
                "projected_pattern": projected.reshape(10, 10, 1, 200, 200),
                "weighted_zero_mean_pattern": weighted_zero_mean.reshape(10, 10, 1, 200, 200),
                "template_support_pattern": support_pattern.reshape(10, 10, 1, 200, 200),
                "normalized_pattern": normalized.reshape(10, 10, 1, 200, 200),
                "command_delta": command_delta.reshape(10, 10, 1, 200, 200),
                "command": rebuilt_command.reshape(10, 10, 1, 200, 200),
                "physical_slm_intensity": intensity.reshape(10, 10, 1, 200, 200),
            }
            for name, values in batch_values.items():
                stages[name].add(code_start, values)
            proto_mask = prototypes[:, 0].index_select(0, label_input).reshape(10, 10, 200, 200) >= 0.2
            delta_flat = command_delta.reshape(10, 10, 200, 200)
            support_energy += (delta_flat.square() * proto_mask).sum().double()
            delta_energy += delta_flat.square().sum().double()
            reconstruction_errors["basis"] = max(reconstruction_errors["basis"], float((projected - torch.einsum("bk,khw->bhw", coefficients, basis)[:, None]).abs().max()))
            reconstruction_errors["weighted_mean"] = max(reconstruction_errors["weighted_mean"], float(((support * projected).flatten(1).sum(1, keepdim=True) / support.flatten(1).sum(1, keepdim=True).clamp_min(1.0e-8) - weighted_mean).abs().max()))
            reconstruction_errors["normalization"] = max(reconstruction_errors["normalization"], float((normalized * max_abs[:, :, None, None] - support_pattern).abs().max()))
            reconstruction_errors["command"] = max(reconstruction_errors["command"], float((rebuilt_command - command).abs().max()))
    stage_reports = {name: stage.report() for name, stage in stages.items()}
    transition_order = (
        "projected_pattern", "weighted_zero_mean_pattern", "template_support_pattern",
        "normalized_pattern", "command_delta", "command", "physical_slm_intensity",
    )
    transition_metrics = {}
    for source, target in zip(transition_order, transition_order[1:]):
        source_stage = stage_reports[source]
        target_stage = stage_reports[target]
        transition_metrics[f"{source}_to_{target}"] = {
            "code_ss_retention": target_stage["code_effect_ss"] / max(source_stage["code_effect_ss"], 1.0e-12),
            "label_ss_retention": target_stage["label_effect_ss"] / max(source_stage["label_effect_ss"], 1.0e-12),
            "factor_retention": target_stage["code_fraction"] / max(source_stage["code_fraction"], 1.0e-12),
        }
    report = {
        "schema": "v781_semantic_command_bottleneck_v1", "status": "v781-semantic-command-bottleneck-complete", "diagnostic_only": True,
        "checkpoint": str(checkpoint), "config": str(config), "validation_dataset": str(validation_dataset),
        "validation_sample_count": CODE_COUNT, "crossed_output_count": CODE_COUNT * LABEL_COUNT, "test_split_used": False,
        "return_optical_fields": False, "persistent_full_field_saved": False, "saved_artifacts": "scalar_statistics_only",
        "semantic_amplitude": AMPLITUDE, "checkpoint_boundary_contract": checkpoint_boundary_contract(payload),
        "checkpoint_model_hash": before_state_hash, "parameter_hash_before": before_state_hash,
        "parameter_hash_after": _state_hash(model),
        "state_keys_equal": before_keys == tuple(sorted(model.state_dict())),
        "codes_hash": tensor_hash(codes), "prototype_hash": tensor_hash(prototypes),
        "stages": stage_reports,
        "transition_metrics": transition_metrics,
        "prototype_support_command_delta_energy_fraction": float(support_energy / delta_energy.clamp_min(1.0e-12)),
        "reconstruction_errors": reconstruction_errors,
        "parameters_unchanged": before_state_hash == _state_hash(model),
        "optimizer_created": False, "parameter_update_count": 0,
        "finite": all(item["finite"] for item in (stage.report() for stage in stages.values())),
        "v78_status": decision.get("status"),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--v78-admission", required=True)
    parser.add_argument("--v78-decision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
