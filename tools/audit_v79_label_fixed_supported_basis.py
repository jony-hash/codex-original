"""Generator-only v79 label-fixed supported-basis upper-bound audit."""

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
V73_SLM_CODE_SS = 289721.29211025406
V73_SLM_LABEL_SS = 757421.0198058158


class _StreamingStage:
    def __init__(self) -> None:
        self.code_sum = None
        self.label_sum = None
        self.total_sum = None
        self.total_sq = None
        self.count = 0
        self.digest = hashlib.sha256()

    def add(self, values: torch.Tensor) -> None:
        if values.ndim < 3 or values.shape[:2] != (10, 10):
            raise ValueError("v79 chunks must have shape [10,10,...]")
        flat = values.detach().float().cpu().reshape(10, 10, -1).double()
        code_sum = flat.sum(1)
        label_sum = flat.sum(0)
        total_sum = flat.sum((0, 1))
        total_sq = flat.square().sum((0, 1))
        if self.code_sum is None:
            self.code_sum = torch.zeros(CODE_COUNT, flat.shape[-1], dtype=torch.float64)
            self.label_sum = torch.zeros(LABEL_COUNT, flat.shape[-1], dtype=torch.float64)
            self.total_sum = torch.zeros(flat.shape[-1], dtype=torch.float64)
            self.total_sq = torch.zeros(flat.shape[-1], dtype=torch.float64)
        self.code_sum[self.count // LABEL_COUNT : self.count // LABEL_COUNT + 10] += code_sum
        self.label_sum += label_sum
        self.total_sum += total_sum
        self.total_sq += total_sq
        self.count += 100
        self.digest.update(values.detach().float().cpu().contiguous().numpy().tobytes())

    def report(self) -> dict[str, Any]:
        if self.code_sum is None or self.label_sum is None or self.total_sum is None or self.total_sq is None:
            raise ValueError("empty v79 stage")
        grand = self.total_sum / float(self.count)
        code_mean = self.code_sum / float(LABEL_COUNT)
        label_mean = self.label_sum / float(CODE_COUNT)
        total_ss = float((self.total_sq - self.count * grand.square()).sum())
        code_ss = float(LABEL_COUNT * (code_mean - grand).square().sum())
        label_ss = float(CODE_COUNT * (label_mean - grand).square().sum())
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
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _low_frequency_fraction(values: torch.Tensor) -> float:
    centered = values.float() - values.float().mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fftshift(torch.fft.fft2(centered), dim=(-2, -1)).abs().square()
    fy = torch.fft.fftshift(torch.fft.fftfreq(values.shape[-2], d=2.0e-6)) / 1000.0
    fx = torch.fft.fftshift(torch.fft.fftfreq(values.shape[-1], d=2.0e-6)) / 1000.0
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    low = spectrum[..., radius <= 20.0].sum()
    return float((low / spectrum.sum().clamp_min(1.0e-12)).item())


def _prepare(cfg: Any, checkpoint: str | Path, validation_dataset: str | Path) -> tuple[OneShotOpticalGenerator, dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes = data["codes"].float()
    labels = data["class_labels"].long()
    if tuple(codes.shape) != (CODE_COUNT, 128) or tuple(labels.shape) != (CODE_COUNT,) or set(labels.tolist()) != set(range(LABEL_COUNT)):
        raise ValueError("v79 requires the fixed validation split")
    prototypes = _build_prototype_bank(cfg).float().contiguous()
    model = OneShotOpticalGenerator(cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size).eval()
    load_one_shot_checkpoint(model, payload)
    return model, payload, codes, labels, prototypes


def _arm_metrics(model: OneShotOpticalGenerator, codes: torch.Tensor, prototypes: torch.Tensor, *, capacity: bool, basis: torch.Tensor, psi_fixed: torch.Tensor, base_command: torch.Tensor) -> dict[str, Any]:
    stage = _StreamingStage()
    support_energy = torch.zeros((), dtype=torch.float64)
    delta_energy = torch.zeros((), dtype=torch.float64)
    low_frequency = []
    max_q = 0.0
    clipping_count = 0
    pixel_count = 0
    formula_errors = {"basis": 0.0, "weighted_mean": 0.0, "q": 0.0, "command": 0.0, "intensity": 0.0}
    with torch.no_grad():
        for start in range(0, CODE_COUNT, 10):
            code_chunk = codes[start : start + 10]
            code_input = code_chunk.repeat_interleave(LABEL_COUNT, dim=0)
            label_input = torch.arange(LABEL_COUNT).repeat(code_chunk.shape[0])
            if capacity:
                coefficients = torch.tanh(code_input[:, :8])
            else:
                residual = model.generator_e.residual_net(model.generator_e._features(code_input, label_input)).reshape(-1, 1, 200, 200)
                _, coefficients = project_residual_to_basis(residual, basis)
            coefficients = coefficients.reshape(10, 10, 8)
            q = torch.einsum("clk,lkhw->clhw", coefficients, psi_fixed).unsqueeze(2)
            command_delta = AMPLITUDE * q
            command = (base_command[None, :, None] + command_delta).clamp(0.0, 1.0)
            intensity = (BACKGROUND_FLOOR + (1.0 - BACKGROUND_FLOOR) * command).square()
            stage.add(intensity)
            low_frequency.append(_low_frequency_fraction(command_delta))
            max_q = max(max_q, float(q.abs().max()))
            clipping_count += int(((command <= 0.0) | (command >= 1.0)).sum().item())
            pixel_count += int(command.numel())
            support_mask = (prototypes[:, 0] >= 0.2)[None, :, None]
            support_energy += (command_delta.square() * support_mask).sum().double()
            delta_energy += command_delta.square().sum().double()
            expected_q = torch.einsum("clk,lkhw->clhw", coefficients, psi_fixed).unsqueeze(2)
            formula_errors["q"] = max(formula_errors["q"], float((expected_q - q).abs().max()))
            formula_errors["command"] = max(formula_errors["command"], float(((base_command[None, :, None] + AMPLITUDE * expected_q).clamp(0.0, 1.0) - command).abs().max()))
            formula_errors["intensity"] = max(formula_errors["intensity"], float((((BACKGROUND_FLOOR + (1.0 - BACKGROUND_FLOOR) * command).square() - intensity).abs()).max()))
    report = stage.report()
    return {
        **report,
        "capacity_arm": capacity,
        "prototype_support_command_delta_energy_fraction": float(support_energy / delta_energy.clamp_min(1.0e-12)),
        "low_frequency_energy_fraction": float(sum(low_frequency) / len(low_frequency)),
        "one_clipping_fraction": clipping_count / max(pixel_count, 1),
        "max_abs_q": max_q,
        "formula_errors": formula_errors,
    }


def run(*, config: str | Path, checkpoint: str | Path, validation_dataset: str | Path, v73_report: str | Path, v781_decision: str | Path, output: str | Path) -> dict[str, Any]:
    cfg = load_training_config(config)
    v73 = _read(v73_report)
    v781 = _read(v781_decision)
    if v781.get("status") != "mixed-support-normalization-code-suppression":
        raise ValueError("v79 requires the frozen v78.1 mixed suppression decision")
    if v73.get("status") != "v73-semantic-support-audit-complete" or v73.get("test_split_used"):
        raise ValueError("v79 requires the valid v73 validation report")
    model, payload, codes, labels, prototypes = _prepare(cfg, checkpoint, validation_dataset)
    before_hash = _state_hash(model)
    before_keys = tuple(sorted(model.state_dict()))
    basis = semantic_basis(200)
    template = torch.sigmoid(model.generator_e.template_logits.detach())[:, 0]
    base_command = ((template.sqrt() - BACKGROUND_FLOOR) / (1.0 - BACKGROUND_FLOOR)).clamp(0.0, 1.0)
    weighted_mu = (template[:, None] * basis[None]).flatten(2).sum(2) / template.flatten(1).sum(1)[:, None].clamp_min(1.0e-8)
    psi = template[:, None] * (basis[None] - weighted_mu[:, :, None, None])
    normalizer = psi.abs().sum(1).flatten(1).amax(1).clamp_min(1.0e-8)
    psi_fixed = psi / normalizer[:, None, None, None]
    learned = _arm_metrics(model, codes, prototypes, capacity=False, basis=basis, psi_fixed=psi_fixed, base_command=base_command)
    capacity = _arm_metrics(model, codes, prototypes, capacity=True, basis=basis, psi_fixed=psi_fixed, base_command=base_command)
    psi_rebuilt = template[:, None] * (basis[None] - weighted_mu[:, :, None, None])
    normalized_rebuilt = psi_rebuilt / normalizer[:, None, None, None]
    formula_errors = {
        "basis_reconstruction": float((psi_rebuilt - template[:, None] * (basis[None] - weighted_mu[:, :, None, None])).abs().max()),
        "weighted_mean": float((((template[:, None] * basis[None]).sum((-2, -1)) / template.sum((-2, -1))[:, None].clamp_min(1.0e-8)) - weighted_mu).abs().max()),
        "normalization": float((normalized_rebuilt - psi_fixed).abs().max()),
    }
    learned["formula_errors"].update(formula_errors)
    capacity["formula_errors"].update(formula_errors)
    after_hash = _state_hash(model)
    state_keys_equal = before_keys == tuple(sorted(model.state_dict()))
    report = {
        "schema": "v79_label_fixed_supported_basis_audit_v1",
        "status": "v79-label-fixed-supported-basis-audit-complete",
        "diagnostic_only": True,
        "checkpoint": str(checkpoint),
        "config": str(config),
        "validation_dataset": str(validation_dataset),
        "v73_report": str(v73_report),
        "v781_decision": str(v781_decision),
        "validation_sample_count": CODE_COUNT,
        "crossed_output_count": CODE_COUNT * LABEL_COUNT,
        "test_split_used": False,
        "optical_forward_used": False,
        "cuda_used": False,
        "optimizer_created": False,
        "parameter_update_count": 0,
        "parameter_hash_before": before_hash,
        "parameter_hash_after": after_hash,
        "parameters_unchanged": before_hash == after_hash,
        "state_keys_equal": state_keys_equal,
        "checkpoint_boundary_contract": checkpoint_boundary_contract(payload),
        "checkpoint_state_keys_compatible": set(payload["model_state_dict"]) == set(model.state_dict()),
        "codes_hash": tensor_hash(codes),
        "prototype_hash": tensor_hash(prototypes),
        "basis_hash": tensor_hash(basis),
        "label_normalizer_hash": tensor_hash(normalizer),
        "semantic_amplitude": AMPLITUDE,
        "normalization_contract": "label_only_N_l=max_p_sum_k_abs_psi_lk",
        "arms": {"learned_replay": learned, "deterministic_capacity": capacity},
        "v73_baseline": {"physical_slm_code_effect_ss": V73_SLM_CODE_SS, "physical_slm_label_effect_ss": V73_SLM_LABEL_SS},
        "finite": bool(learned["finite"] and capacity["finite"] and torch.isfinite(normalizer).all()),
        "persistent_full_field_saved": False,
        "saved_artifacts": "scalar_statistics_only",
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
    parser.add_argument("--v73-report", required=True)
    parser.add_argument("--v781-decision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
