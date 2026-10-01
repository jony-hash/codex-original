"""No-update modal-interface and optical-readout attribution.

This module is intentionally separate from the historical ``fiber-attribution``
v1 lifecycle.  It consumes the same synthetic research initialization, but
reissues the upstream physical/evaluator validation at the current commit and
never creates an optimizer or changes a trainable parameter.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from ldm_project.v2.contracts import ComplexField, OperationProvenance
from ldm_project.v2.evidence import atomic_torch_save, atomic_write_json, file_sha256, safe_torch_load
from ldm_project.v2.optics.devices import identity_ols
from ldm_project.v2.optics.fiber_operator import FiberArrayOLS, FiberArraySpec
from ldm_project.v2.optics.propagation import clear_production_transfer_cache
from ldm_project.v2.student.conditional import camera_code_fraction, fixed_exposure_view, relative_diversity_ratios
from ldm_project.v2.student.conditional_acceptance import conditional_coverage_ratio
from ldm_project.v2.application._runtime_shared import random_codes
from ldm_project.v2.application.research.optical_manifold import fiber_attribution as historical
from ldm_project.v2.application.research.optical_manifold import fiber_image_generation as image

PANELS = (4242, 104729, 161803)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
VARIANTS = ("I0", "S0", "L0", "R25", "R50", "R75", "R100", "S0-energy-oracle")
PHYSICAL_VARIANTS = ("S0", "L0", "R25", "R50", "R75", "R100")
CANDIDATE_VARIANTS = ("R25", "R50", "R75", "R100")
STAGES = (
    "slm_output",
    "encoder_input",
    "encoder_layer_1",
    "encoder_layer_2",
    "ols_output",
    "decoder_layer_1",
    "decoder_layer_2",
    "camera",
    "fixed_cmos",
)
STAGE_EMBEDDING_SIZE = 28
STORAGE_LIMIT_BYTES = 2 * 1024**3
JACOBIAN_RANK_THRESHOLD = 1.0e-3
SHUFFLED_LABEL_REPETITIONS = 8
RESIDUAL_SPECTRUM_BANDS = ((0.0, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 1.01))
_SPECTRUM_MASK_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def _write(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _require_file(args: Any, name: str) -> Path:
    value = getattr(args, name, None)
    if not value:
        raise ValueError(f"fiber interface attribution requires --{name.replace('_', '-')}")
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"fiber interface attribution input does not exist: {path}")
    return path


def _git() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _root(output: str | Path, *, require_new: bool = False) -> Path:
    path = Path(output)
    if path.suffix == ".json":
        root = path.parent
    else:
        root = path
    if require_new and root.exists() and any(root.iterdir()):
        raise ValueError(f"interface attribution output must be a new directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tensor_hash(value: torch.Tensor) -> str:
    return image._tensor_content_hash(value)


def _ref(path: Path, root: Path, value: torch.Tensor | None = None) -> dict[str, str]:
    reference = {
        "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": file_sha256(path),
    }
    if value is not None:
        reference["tensor_sha256"] = _tensor_hash(value)
    return reference


def _resolve(reference: dict[str, Any], root: Path) -> Path:
    relative = reference.get("relative_path")
    expected = reference.get("sha256")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("invalid interface attribution artifact path")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file() or file_sha256(path) != expected:
        raise ValueError("missing or hash-mismatched interface attribution artifact")
    return path


def _residual_spectrum_bands(residual: torch.Tensor) -> torch.Tensor:
    height, width = residual.shape[-2:]
    key = (height, width, str(residual.device))
    masks = _SPECTRUM_MASK_CACHE.get(key)
    if masks is None:
        fy = torch.fft.fftshift(torch.fft.fftfreq(height, device=residual.device))
        fx = torch.fft.fftshift(torch.fft.fftfreq(width, device=residual.device))
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square()) / (2.0**0.5 / 2.0)
        masks = torch.stack(
            [
                ((radius >= low) & (radius < high)).to(residual.real.dtype)
                for low, high in RESIDUAL_SPECTRUM_BANDS
            ],
            dim=0,
        )
        _SPECTRUM_MASK_CACHE[key] = masks
    spectrum = torch.fft.fftshift(torch.fft.fft2(residual), dim=(-2, -1)).abs().square()
    total = spectrum.sum(dim=(-2, -1)).clamp_min(1.0e-30)
    return torch.einsum("bhw,khw->bk", spectrum, masks) / total[:, None]


def _gram_summary(modes: torch.Tensor, area: float) -> dict[str, Any]:
    gram = modes.flatten(1).conj() @ modes.flatten(1).T * area
    identity = torch.eye(modes.shape[0], device=gram.device, dtype=gram.dtype)
    singular = torch.linalg.svdvals(gram)
    return {
        "rank": int(torch.linalg.matrix_rank(gram).item()),
        "condition": float(torch.linalg.cond(gram)),
        "gram_identity_max_abs": float((gram - identity).abs().max()),
        "finite": bool(torch.isfinite(gram).all()),
        "shape": list(gram.shape),
        "singular_values": singular[:16].tolist(),
    }


def _lineage_namespace(args: Any, root: Path) -> SimpleNamespace:
    values = vars(args).copy()
    values["measurement_bundle"] = str(root / "fiber_image_measurement_bundle.json")
    values["measurement_evidence"] = str(root / "fiber_image_measurement_evidence.json")
    values["validate_evidence"] = str(root / "fiber_image_validation.json")
    values["output"] = str(root / "fiber_image_validation.json")
    return SimpleNamespace(**values)


def _validate_upstream(args: Any, root: Path) -> dict[str, Any]:
    upstream = _lineage_namespace(args, root)
    upstream.measurement_bundle_output = upstream.measurement_bundle
    upstream.measurement_evidence_output = upstream.measurement_evidence
    image.validate(upstream)
    payload = json.loads(Path(upstream.output).read_text(encoding="utf-8"))
    if payload.get("schema") != "fiber_image_validation.v2" or payload.get("gate") != "valid":
        raise ValueError("upstream fiber image validation failed")
    return payload


def _validate_historical_inputs(args: Any, upstream: dict[str, Any]) -> dict[str, Any]:
    v3_path = _require_file(args, "v3_closeout")
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    if (
        v3.get("schema") != "fiber_image_closeout.v2"
        or v3.get("gate") != "valid_failed"
        or v3.get("conclusion") != "no_reproducible_fiber_gain"
        or v3.get("formal_promotion") is not False
    ):
        raise ValueError("interface attribution requires the recorded negative v3 closeout")
    attribution_path = _require_file(args, "attribution_measurement")
    attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
    if (
        attribution.get("schema") != "fiber_attribution_measurement.v1"
        or attribution.get("gate") != "valid"
        or attribution.get("status") != "synthetic diagnostic-only"
        or tuple(attribution.get("variants", ())) != historical.VARIANTS
        or tuple(attribution.get("panels", ())) != historical.PANELS
    ):
        raise ValueError("historical attribution measurement is not a valid bound input")
    return {
        "v3_closeout_sha256": file_sha256(v3_path),
        "attribution_measurement_sha256": file_sha256(attribution_path),
        "lp01_fixture_sha256": file_sha256(_require_file(args, "lp01_fixture")),
        "research_initialization_checkpoint_sha256": file_sha256(
            _require_file(args, "research_initialization_checkpoint")
        ),
        "source_training_evidence_sha256": file_sha256(
            _require_file(args, "source_training_evidence")
        ),
        "v3_closeout": v3,
        "attribution_measurement": attribution,
        "upstream": upstream,
    }


def _context(args: Any, root: Path) -> tuple[Any, ...]:
    context_args = _lineage_namespace(args, root)
    return image._context(context_args)


class _ResidualBlendOLS(nn.Module):
    """Synthetic projection plus orthogonal residual interpolation."""

    diagnostic_only = True

    def __init__(self, input_plane: Any, *, alpha: float, energy_oracle: bool = False) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("residual interpolation alpha must be in [0, 1]")
        self.fiber = FiberArrayOLS(input_plane, FiberArraySpec())
        self.alpha = float(alpha)
        self.energy_oracle = bool(energy_oracle)
        self.last_diagnostics: dict[str, Any] = {}
        self.last_port_coefficients: torch.Tensor | None = None

    def _coefficients(self, field: ComplexField) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = field.tensor[:, 0] if field.tensor.ndim == 4 else field.tensor
        modes = self.fiber.input_modes.to(values)
        area = self.fiber.input_plane.pixel_pitch_m[0] * self.fiber.input_plane.pixel_pitch_m[1]
        coefficients = torch.einsum("mhw,bhw->bm", modes.conj(), values) * area
        output_coefficients = coefficients @ self.fiber.transport.to(values).T
        output_values = torch.einsum(
            "bm,mhw->bhw", output_coefficients, self.fiber.output_modes.to(values)
        )
        return values, output_values, output_coefficients

    def forward(self, field: ComplexField) -> ComplexField:
        values, projected, output_coefficients = self._coefficients(field)
        residual = values - projected
        if self.energy_oracle:
            area = self.fiber.input_plane.pixel_pitch_m[0] * self.fiber.input_plane.pixel_pitch_m[1]
            input_energy = values.abs().square().sum(dim=(-2, -1)) * area
            projected_energy = projected.abs().square().sum(dim=(-2, -1)) * area
            scale = (input_energy / projected_energy.clamp_min(1.0e-30)).sqrt()
            output = projected * scale[:, None, None]
            alpha = 0.0
        else:
            output = projected + self.alpha * residual
            alpha = self.alpha
        area = self.fiber.input_plane.pixel_pitch_m[0] * self.fiber.input_plane.pixel_pitch_m[1]
        input_energy = values.abs().square().sum(dim=(-2, -1)) * area
        output_energy = output.abs().square().sum(dim=(-2, -1)) * area
        projected_energy = projected.abs().square().sum(dim=(-2, -1)) * area
        residual_energy = residual.abs().square().sum(dim=(-2, -1)) * area
        if not self.energy_oracle and torch.any(output_energy > input_energy * (1.0 + 1.0e-5) + 1.0e-12):
            raise ValueError("residual interpolation violates the passive energy gate")
        self.last_port_coefficients = output_coefficients
        self.last_diagnostics = {
            "input_energy": input_energy.detach(),
            "output_energy": output_energy.detach(),
            "projected_energy": projected_energy.detach(),
            "residual_energy": residual_energy.detach(),
            "projection_fraction": (projected_energy / input_energy.clamp_min(1.0e-30)).detach(),
            "residual_fraction": (residual_energy / input_energy.clamp_min(1.0e-30)).detach(),
            "port_input_coefficients": (torch.einsum(
                "mhw,bhw->bm", self.fiber.input_modes.to(values).conj(), values
            ) * area).detach(),
            "port_output_coefficients": output_coefficients.detach(),
            "residual_spectrum_bands": (
                _residual_spectrum_bands(residual).detach()
                if not torch.is_grad_enabled()
                else None
            ),
            "passive": bool(torch.all(output_energy <= input_energy * (1.0 + 1.0e-5) + 1.0e-12)),
            "energy_oracle": self.energy_oracle,
            "alpha": alpha,
        }
        return ComplexField(
            output.unsqueeze(1) if field.tensor.ndim == 4 else output,
            self.fiber.input_plane.with_plane("ols_output"),
            OperationProvenance(
                "synthetic_smf_lp01_residual_blend",
                self.fiber.input_plane.plane_id,
                0.0,
                {"alpha": alpha, "energy_oracle": self.energy_oracle, "physical_parameter_status": "synthetic diagnostic-only"},
            ),
        )


class _IdentityOLS(nn.Module):
    diagnostic_only = True

    def forward(self, field: ComplexField) -> ComplexField:
        return identity_ols(field)


def _install_variant(bundle: Any, variant: str, device: torch.device) -> nn.Module | _IdentityOLS:
    if variant == "I0":
        operator = _IdentityOLS()
    elif variant in {"S0", "L0"}:
        operator = _ResidualBlendOLS(bundle.system.encoder_plane, alpha=0.0)
    elif variant == "S0-energy-oracle":
        operator = _ResidualBlendOLS(bundle.system.encoder_plane, alpha=0.0, energy_oracle=True)
    elif variant.startswith("R"):
        alpha = int(variant[1:]) / 100.0
        operator = _ResidualBlendOLS(bundle.system.encoder_plane, alpha=alpha)
    else:
        raise ValueError(f"unknown interface attribution variant: {variant}")
    bundle.system.ols = operator.to(device)
    return operator


def _stage_embedding(value: torch.Tensor, *, intensity: bool = False) -> torch.Tensor:
    if intensity:
        return image._deterministic_adaptive_average(value, STAGE_EMBEDDING_SIZE).flatten(1)
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError("stage field must have shape [N,1,H,W]")
    real = image._deterministic_adaptive_average(value.real, STAGE_EMBEDDING_SIZE)
    imag = image._deterministic_adaptive_average(value.imag, STAGE_EMBEDDING_SIZE)
    return torch.cat((real, imag), dim=1).flatten(1)


def _nearest_centroid(train: torch.Tensor, train_labels: torch.Tensor, test: torch.Tensor, test_labels: torch.Tensor, *, unit: bool) -> float:
    if unit:
        train = torch.nn.functional.normalize(train, dim=1)
        test = torch.nn.functional.normalize(test, dim=1)
    centroids = torch.stack([train[train_labels == label].mean(0) for label in range(10)])
    distances = test.square().sum(1, keepdim=True) + centroids.square().sum(1).unsqueeze(0) - 2.0 * test @ centroids.T
    return float((distances.argmin(1) == test_labels).float().mean())


def _cross_panel_nearest(embeddings: dict[str, dict[int, torch.Tensor]], labels: dict[int, torch.Tensor]) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for variant_index, (variant, stage_rows) in enumerate(embeddings.items()):
        result[variant] = {}
        for stage_index, (stage, panel_rows) in enumerate(stage_rows.items()):
            scores = {"raw": [], "unit": [], "shuffled_raw": [], "shuffled_unit": []}
            for test_panel in PANELS:
                train_panels = [panel for panel in PANELS if panel != test_panel]
                train = torch.cat([panel_rows[panel] for panel in train_panels])
                train_labels = torch.cat([labels[panel] for panel in train_panels])
                test = panel_rows[test_panel]
                test_labels = labels[test_panel]
                scores["raw"].append(_nearest_centroid(train, train_labels, test, test_labels, unit=False))
                scores["unit"].append(_nearest_centroid(train, train_labels, test, test_labels, unit=True))
                shuffled_raw = []
                shuffled_unit = []
                for shuffle_index in range(SHUFFLED_LABEL_REPETITIONS):
                    shuffle_generator = torch.Generator(device="cpu").manual_seed(
                        701
                        + variant_index * 1000
                        + stage_index * 100
                        + PANELS.index(test_panel) * 10
                        + shuffle_index
                    )
                    shuffled = train_labels[
                        torch.randperm(train_labels.numel(), generator=shuffle_generator)
                    ]
                    shuffled_raw.append(
                        _nearest_centroid(train, shuffled, test, test_labels, unit=False)
                    )
                    shuffled_unit.append(
                        _nearest_centroid(train, shuffled, test, test_labels, unit=True)
                    )
                scores["shuffled_raw"].append(float(sum(shuffled_raw) / len(shuffled_raw)))
                scores["shuffled_unit"].append(float(sum(shuffled_unit) / len(shuffled_unit)))
            result[variant][stage] = {key: float(sum(values) / len(values)) for key, values in scores.items()}
    return result


def _rank_summary(coefficients: torch.Tensor) -> dict[str, Any]:
    values = torch.cat((coefficients.real, coefficients.imag), dim=1).double()
    values = values - values.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(values)
    energy = singular.square()
    total = energy.sum().clamp_min(1.0e-30)
    probabilities = energy / total
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1.0e-30).log()).sum())
    return {
        "effective_rank": float(effective_rank),
        "rank_at_1e-8": int((singular > singular.max().clamp_min(1.0e-30) * 1.0e-8).sum()),
        "top8_energy_fraction": float(energy[:8].sum() / total),
        "top16_energy_fraction": float(energy[:16].sum() / total),
        "singular_values": singular[:16].tolist(),
    }


def _class_geometry(coefficients: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    values = torch.cat((coefficients.real, coefficients.imag), dim=1).float()
    centroids = torch.stack([values[labels == label].mean(0) for label in range(10)])
    per_class_within = torch.stack([
        (values[labels == label] - centroids[label]).square().sum(1).mean().sqrt()
        for label in range(10)
    ])
    within = per_class_within.mean()
    between = torch.pdist(centroids).mean()
    return {
        "centroid_between_over_within": float(between / within.clamp_min(1.0e-30)),
        "within_rms": float(within),
        "between_centroid_mean": float(between),
        "per_class_within_rms": per_class_within.tolist(),
    }


def _relative_l2(current: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (current - reference).abs().square().sum().sqrt()
    denominator = reference.abs().square().sum().sqrt().clamp_min(1.0e-30)
    return float(numerator / denominator)


def _mode_bank_spec_hash() -> str:
    payload = json.dumps(asdict(FiberArraySpec()), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jacobian_point(
    bundle: Any,
    code: torch.Tensor,
    label: int,
    device: torch.device,
    exposure_scale: float,
) -> dict[str, Any]:
    """Measure local code-to-port and code-to-CMOS maps without updates."""
    if code.shape != (8,):
        raise ValueError("Jacobian probe code must have shape [8]")
    label_tensor = torch.tensor([label], device=device, dtype=torch.long)

    def output(value: torch.Tensor) -> torch.Tensor:
        command = bundle.generator(value.unsqueeze(0), label_tensor)
        result = bundle.system.forward(bundle.system.command(command), return_optical_trace=False)
        operator = bundle.system.ols
        if not isinstance(operator, _ResidualBlendOLS) or operator.last_port_coefficients is None:
            raise ValueError("Jacobian probe requires the synthetic residual OLS operator")
        ports = operator.last_port_coefficients[0]
        camera = image._deterministic_fixed_exposure_view(
            result.raw_camera.tensor,
            exposure_scale,
            output_size=28,
        )[0].flatten()
        return torch.cat((ports.real, ports.imag, camera))

    base_code = code.detach().to(device)
    columns = []
    with torch.inference_mode(False), torch.enable_grad():
        for direction in torch.eye(8, device=device):
            _, derivative = torch.autograd.functional.jvp(
                output, base_code, direction, create_graph=False
            )
            columns.append(derivative)
    jacobian = torch.stack(columns, dim=1)
    if jacobian.shape != (984, 8):
        raise ValueError(f"unexpected Jacobian shape: {tuple(jacobian.shape)}")
    port_jacobian = jacobian[:200].double()
    camera_jacobian = jacobian[200:].double()
    port_singular = torch.linalg.svdvals(port_jacobian)
    camera_singular = torch.linalg.svdvals(camera_jacobian)
    port_scale = port_singular.max().clamp_min(1.0e-30)
    camera_scale = camera_singular.max().clamp_min(1.0e-30)
    port_rank_1e3 = int((port_singular > port_scale * JACOBIAN_RANK_THRESHOLD).sum())
    camera_rank_1e3 = int((camera_singular > camera_scale * JACOBIAN_RANK_THRESHOLD).sum())
    port_rank_1e8 = int((port_singular > port_scale * 1.0e-8).sum())
    camera_rank_1e8 = int((camera_singular > camera_scale * 1.0e-8).sum())
    _, right = torch.linalg.eigh(port_jacobian.T @ port_jacobian)
    directions = right[:, -8:]
    port_response = torch.linalg.vector_norm(port_jacobian @ directions, dim=0)
    camera_response = torch.linalg.vector_norm(camera_jacobian @ directions, dim=0)
    normalized_port = port_jacobian / port_jacobian.norm().clamp_min(1.0e-30)
    normalized_camera = camera_jacobian / camera_jacobian.norm().clamp_min(1.0e-30)
    normalized_port_response = torch.linalg.vector_norm(normalized_port @ directions, dim=0)
    normalized_camera_response = torch.linalg.vector_norm(normalized_camera @ directions, dim=0)
    return {
        "label": label,
        "shape": [200, 8],
        "cmos_shape": [784, 8],
        "port_rank_1e-3": port_rank_1e3,
        "cmos_rank_1e-3": camera_rank_1e3,
        "port_rank_1e-8": port_rank_1e8,
        "cmos_rank_1e-8": camera_rank_1e8,
        "port_rank": port_rank_1e3,
        "cmos_rank": camera_rank_1e3,
        "port_to_cmos_rank_loss": max(port_rank_1e3 - camera_rank_1e3, 0),
        "port_condition": float(port_singular.max() / port_singular[-1].clamp_min(1.0e-30)),
        "cmos_condition": float(camera_singular.max() / camera_singular[-1].clamp_min(1.0e-30)),
        "port_singular_values": port_singular.tolist(),
        "cmos_singular_values": camera_singular.tolist(),
        "port_response_by_port_direction": port_response.tolist(),
        "cmos_response_by_port_direction": camera_response.tolist(),
        "port_to_cmos_response_ratio": (
            normalized_camera_response / normalized_port_response.clamp_min(1.0e-30)
        ).tolist(),
        "finite": bool(torch.isfinite(jacobian).all()),
    }


def _jacobian_summary(context: tuple[Any, ...], codes_by_panel: dict[int, torch.Tensor]) -> dict[str, Any]:
    config, device, _, _, _, _, _, _, bundle, _, _, _ = context
    settings = config.conditional_generation
    if settings is None:
        raise ValueError("conditional generation settings are required")
    _install_variant(bundle, "S0", device)
    clear_production_transfer_cache()
    points = []
    for panel in PANELS:
        codes = codes_by_panel[panel]
        for label in range(10):
            point = _jacobian_point(
                bundle, codes[label * 50], label, device, settings.fixed_exposure_scale
            )
            point["panel"] = panel
            points.append(point)
    port_ranks = [point["port_rank"] for point in points]
    cmos_ranks = [point["cmos_rank"] for point in points]
    losses = [ratio for point in points for ratio in point["port_to_cmos_response_ratio"]]
    return {
        "points": points,
        "point_count": len(points),
        "port_rank_mean": float(sum(port_ranks) / len(port_ranks)),
        "port_rank_min": min(port_ranks),
        "cmos_rank_mean": float(sum(cmos_ranks) / len(cmos_ranks)),
        "port_to_cmos_response_ratio_mean": float(sum(losses) / len(losses)),
        "finite": all(point["finite"] for point in points),
    }


def _equivalence_probe(bundle: Any, device: torch.device) -> dict[str, float]:
    generator = torch.Generator(device=device).manual_seed(221)
    real = torch.randn((2, 1, *bundle.system.encoder_plane.shape), generator=generator, device=device)
    imag = torch.randn((2, 1, *bundle.system.encoder_plane.shape), generator=generator, device=device)
    field = ComplexField(
        torch.complex(real, imag),
        bundle.system.encoder_plane,
        OperationProvenance("interface_equivalence_probe", None, None),
    )
    s0_a = _ResidualBlendOLS(bundle.system.encoder_plane, alpha=0.0).to(device)(field)
    s0_b = _ResidualBlendOLS(bundle.system.encoder_plane, alpha=0.0).to(device)(field)
    r100 = _ResidualBlendOLS(bundle.system.encoder_plane, alpha=1.0).to(device)(field)
    identity = identity_ols(field)
    return {
        "r0_vs_s0_complex_l2": _relative_l2(s0_a.tensor, s0_b.tensor),
        "r100_vs_i0_complex_l2": _relative_l2(r100.tensor, identity.tensor),
    }


def _assert_validation_lineage(args: Any, validation: dict[str, Any]) -> None:
    expected = {
        "v3_closeout_sha256": file_sha256(_require_file(args, "v3_closeout")),
        "attribution_measurement_sha256": file_sha256(
            _require_file(args, "attribution_measurement")
        ),
        "lp01_fixture_sha256": file_sha256(_require_file(args, "lp01_fixture")),
        "research_initialization_checkpoint_sha256": file_sha256(
            _require_file(args, "research_initialization_checkpoint")
        ),
        "source_training_evidence_sha256": file_sha256(
            _require_file(args, "source_training_evidence")
        ),
        "primary_evaluator_sha256": file_sha256(
            _require_file(args, "primary_evaluator_checkpoint")
        ),
        "secondary_evaluator_sha256": file_sha256(
            _require_file(args, "secondary_evaluator_checkpoint")
        ),
    }
    if any(validation.get(key) != value for key, value in expected.items()):
        raise ValueError("interface validation lineage does not match current inputs")


def _metric_summary(raw: torch.Tensor, view: torch.Tensor, labels: torch.Tensor, primary: Any, secondary: Any, reference_images: torch.Tensor, reference_labels: torch.Tensor, settings: Any) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    model_view = view.to(reference_images.device)
    with torch.inference_mode():
        _, primary_logits = primary(model_view)
        features, secondary_logits = secondary(model_view)
        reference_indices = torch.cat([torch.where(reference_labels == label)[0][:50] for label in range(10)])
        reference_features, _ = secondary(reference_images.index_select(0, reference_indices))
    primary_prediction = primary_logits.argmax(1).cpu()
    secondary_prediction = secondary_logits.argmax(1).cpu()
    labels_cpu = labels.cpu()
    primary_recall = [
        float((primary_prediction[labels_cpu == label] == label).float().mean())
        for label in range(10)
    ]
    secondary_recall = [
        float((secondary_prediction[labels_cpu == label] == label).float().mean())
        for label in range(10)
    ]
    return {
        "primary_accuracy": float((primary_prediction == labels_cpu).float().mean()),
        "secondary_accuracy": float((secondary_prediction == labels_cpu).float().mean()),
        "primary_recall": primary_recall,
        "secondary_recall": secondary_recall,
        "primary_min_recall": min(primary_recall),
        "secondary_min_recall": min(secondary_recall),
        "secondary_prediction_histogram": torch.bincount(secondary_prediction, minlength=10).tolist(),
        "agreement": float((primary_prediction == secondary_prediction).float().mean()),
        "finite": bool(torch.isfinite(raw).all()),
        "nonnegative": bool((raw >= 0).all()),
        "saturation": float((raw > settings.fixed_exposure_scale).float().mean()),
        "camera_fraction": float(camera_code_fraction(raw, labels_cpu).item()),
        "minimum_diversity": float(min(relative_diversity_ratios(features.detach().cpu(), reference_features.detach().cpu(), labels_cpu))),
        "coverage_ratio": float(conditional_coverage_ratio(features, model_view, reference_features, reference_images.index_select(0, reference_indices), labels, epsilon=settings.sinkhorn_epsilon, iterations=settings.sinkhorn_iterations, seed=2027)),
        "labels_sha256": _tensor_hash(labels_cpu),
    }, primary_prediction, secondary_prediction


def _save_variant_artifacts(root: Path, variant: str, panel: int, values: dict[str, Any]) -> dict[str, Any]:
    directory = root / "measurements" / variant / str(panel)
    directory.mkdir(parents=True, exist_ok=True)
    refs: dict[str, Any] = {}
    for name in ("labels", "codes", "fixed_cmos", "primary_prediction", "secondary_prediction", "secondary_correctness"):
        path = directory / f"{name}.pt"
        atomic_torch_save(path, values[name])
        refs[name] = _ref(path, root, values[name])
    trace_path = directory / "representative_trace.pt"
    atomic_torch_save(trace_path, values["representative_trace"])
    refs["representative_trace"] = _ref(trace_path, root)
    if values.get("port_input_coefficients") is not None:
        for name in ("port_input_coefficients", "port_output_coefficients"):
            path = directory / f"{name}.pt"
            atomic_torch_save(path, values[name])
            refs[name] = _ref(path, root, values[name])
    return refs


def _run_variant(context: tuple[Any, ...], variant: str, panel: int, root: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    config, device, _, _, reference_images, reference_labels, primary, secondary, bundle, _, _, _ = context
    settings = config.conditional_generation
    if settings is None:
        raise ValueError("conditional generation settings are required")
    labels = torch.arange(10, device=device).repeat_interleave(50)
    codes = random_codes(500, device=device, seed=panel)
    operator = _install_variant(bundle, variant, device)
    raw_parts: list[torch.Tensor] = []
    view_parts: list[torch.Tensor] = []
    primary_parts: list[torch.Tensor] = []
    secondary_parts: list[torch.Tensor] = []
    stage_parts: dict[str, list[torch.Tensor]] = {stage: [] for stage in STAGES[:-1]}
    input_energy: list[torch.Tensor] = []
    output_energy: list[torch.Tensor] = []
    projection_fraction: list[torch.Tensor] = []
    residual_fraction: list[torch.Tensor] = []
    residual_spectrum: list[torch.Tensor] = []
    port_inputs: list[torch.Tensor] = []
    port_outputs: list[torch.Tensor] = []
    for start in range(0, 500, config.execution.evaluation_microbatch_size):
        c = codes[start : start + config.execution.evaluation_microbatch_size]
        y = labels[start : start + c.shape[0]]
        with torch.inference_mode():
            command = bundle.generator(c, y)
            result = bundle.system.forward(bundle.system.command(command), return_optical_trace=True)
            raw = result.raw_camera.tensor
            view = fixed_exposure_view(raw, settings.fixed_exposure_scale)
            _, p = primary(view)
            _, s = secondary(view)
        raw_parts.append(raw.detach().cpu())
        view_parts.append(view.detach().cpu())
        primary_parts.append(p.detach().cpu())
        secondary_parts.append(s.detach().cpu())
        fields = result.optical_trace or ()
        if len(fields) != len(STAGES) - 1:
            raise ValueError(f"unexpected optical trace length for {variant}: {len(fields)}")
        for stage, field in zip(STAGES[:-1], fields, strict=True):
            stage_parts[stage].append(field.tensor.detach().cpu())
        if variant in PHYSICAL_VARIANTS or variant == "S0-energy-oracle":
            diagnostics = operator.last_diagnostics
            input_energy.append(diagnostics["input_energy"].cpu())
            output_energy.append(diagnostics["output_energy"].cpu())
            projection_fraction.append(diagnostics["projection_fraction"].cpu())
            residual_fraction.append(diagnostics["residual_fraction"].cpu())
            if diagnostics["residual_spectrum_bands"] is not None:
                residual_spectrum.append(diagnostics["residual_spectrum_bands"].cpu())
            port_inputs.append(diagnostics["port_input_coefficients"].cpu())
            port_outputs.append(diagnostics["port_output_coefficients"].cpu())
    raw = torch.cat(raw_parts)
    view = torch.cat(view_parts)
    primary_prediction = torch.cat(primary_parts).argmax(1)
    secondary_prediction = torch.cat(secondary_parts).argmax(1)
    stage_tensors = {stage: torch.cat(values) for stage, values in stage_parts.items()}
    port_input = torch.cat(port_inputs) if port_inputs else None
    port_output = torch.cat(port_outputs) if port_outputs else None
    metric, primary_prediction, secondary_prediction = _metric_summary(raw, view, labels, primary, secondary, reference_images, reference_labels, settings)
    metric.update({
        "variant": variant,
        "panel": panel,
        "codes_sha256": _tensor_hash(codes.cpu()),
        "passive_available": variant in PHYSICAL_VARIANTS,
        "energy_oracle": variant == "S0-energy-oracle",
        "nonphysical_upper_bound": variant == "S0-energy-oracle",
    })
    if input_energy:
        input_e = torch.cat(input_energy)
        output_e = torch.cat(output_energy)
        metric.update({
            "passive": bool(torch.all(output_e <= input_e * (1.0 + 1.0e-5) + 1.0e-12)) if variant in PHYSICAL_VARIANTS else False,
            "input_energy_mean": float(input_e.mean()),
            "output_energy_mean": float(output_e.mean()),
            "projection_fraction_mean": float(torch.cat(projection_fraction).mean()),
            "residual_fraction_mean": float(torch.cat(residual_fraction).mean()),
            "energy_retention_mean": float((output_e / input_e.clamp_min(1.0e-30)).mean()),
        })
        projected_e = input_e * torch.cat(projection_fraction)
        residual_e = input_e * torch.cat(residual_fraction)
        metric["energy_closure_error_max"] = float(
            ((input_e - projected_e - residual_e).abs() / input_e.clamp_min(1.0e-30)).max()
        )
        metric["projection_fraction_per_class"] = [
            float(torch.cat(projection_fraction)[labels.cpu() == label].mean())
            for label in range(10)
        ]
        metric["residual_fraction_per_class"] = [
            float(torch.cat(residual_fraction)[labels.cpu() == label].mean())
            for label in range(10)
        ]
        if residual_spectrum:
            metric["residual_spectrum_bands_mean"] = torch.cat(residual_spectrum).mean(0).tolist()
            metric["residual_spectrum_band_edges"] = [list(edge) for edge in RESIDUAL_SPECTRUM_BANDS]
    if port_input is not None:
        metric["port_rank"] = _rank_summary(port_input)
        metric["port_geometry"] = _class_geometry(port_input, labels.cpu())
    representative = torch.cat([torch.where(labels.cpu() == label)[0][:2] for label in range(10)])
    with torch.inference_mode():
        command = bundle.generator(codes.index_select(0, representative.to(device)), labels.index_select(0, representative.to(device)))
        trace_result = bundle.system.forward(bundle.system.command(command), return_optical_trace=True)
    representative_trace = [
        {
            "tensor": field.tensor.detach().cpu(),
            "plane": asdict(field.plane),
            "provenance": asdict(field.provenance),
        }
        for field in (trace_result.optical_trace or ())
    ]
    stage_energy = {
        stage: float(value.abs().square().sum(dim=(-3, -2, -1)).mean())
        for stage, value in stage_tensors.items()
    }
    metric["stage_energy_mean"] = stage_energy
    stage_order = list(STAGES[:-1])
    metric["stage_energy_retention"] = {
        f"{previous}_to_{current}": stage_energy[current] / max(stage_energy[previous], 1.0e-30)
        for previous, current in zip(stage_order, stage_order[1:])
    }
    metric["stage_finite"] = all(bool(torch.isfinite(value).all()) for value in stage_tensors.values())
    values = {
        **metric,
        "labels": labels.cpu(),
        "codes": codes.cpu(),
        "fixed_cmos": view.cpu(),
        "primary_prediction": primary_prediction,
        "secondary_prediction": secondary_prediction,
        "secondary_correctness": secondary_prediction == labels.cpu(),
        "representative_trace": representative_trace,
        "port_input_coefficients": port_input,
        "port_output_coefficients": port_output,
    }
    embeddings = {stage: _stage_embedding(value) for stage, value in stage_tensors.items()}
    embeddings["fixed_cmos"] = view.cpu().flatten(1)
    return metric, stage_tensors, embeddings, labels.cpu(), values


def _stage_l2(current: dict[str, torch.Tensor], identity: dict[str, torch.Tensor]) -> dict[str, float]:
    result = {}
    for stage in STAGES[:-1]:
        numerator = (current[stage] - identity[stage]).abs().square().sum().sqrt()
        denominator = identity[stage].abs().square().sum().sqrt().clamp_min(1.0e-30)
        result[stage] = float(numerator / denominator)
    return result


def validate(args: Any) -> int:
    output = Path(args.output)
    root = _root(output, require_new=True)
    if _dirty():
        _write(output, {"schema": "fiber_interface_validation.v1", "gate": "invalid_harness", "reason": "dirty worktree"})
        return 2
    try:
        upstream = _validate_upstream(args, root)
        bound = _validate_historical_inputs(args, upstream)
        context = _context(args, root)
        config, device, _, _, _, _, primary, secondary, bundle, _, _, lineage = context
        operator = bundle.system.ols
        if not isinstance(operator, image.FiberTransportEnsembleOLS):
            raise ValueError("research context did not install the expected seeded fiber operator")
        checkpoint = safe_torch_load(
            _require_file(args, "research_initialization_checkpoint"),
            expected_sha256=file_sha256(_require_file(args, "research_initialization_checkpoint")),
            map_location="cpu",
        )
        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("research initialization state_dict is missing")
        if any(not isinstance(key, str) for key in state_dict):
            raise ValueError("research initialization state keys must be strings")
        source_state_keys = sorted(state_dict)
        loaded_generator_keys = sorted(
            key for key in state_dict if isinstance(key, str) and key.startswith("generator.")
        )
        ignored_state_keys = sorted(key for key in state_dict if key not in loaded_generator_keys)
        if not loaded_generator_keys or any(not isinstance(key, str) for key in state_dict):
            raise ValueError("research initialization has no valid generator.* state")
        source_encoder_phase = state_dict.get("system.encoder.phase")
        source_decoder_phase = state_dict.get("system.decoder.phase")
        if not isinstance(source_encoder_phase, torch.Tensor) or not isinstance(source_decoder_phase, torch.Tensor):
            raise ValueError("research initialization must expose source encoder/decoder phase state")
        if not torch.isfinite(source_encoder_phase).all() or not torch.isfinite(source_decoder_phase).all():
            raise ValueError("source optical phase state is non-finite")
        encoder_phase = bundle.system.encoder.phase.detach()
        decoder_phase = bundle.system.decoder.phase.detach()
        area = operator.input_plane.pixel_pitch_m[0] * operator.input_plane.pixel_pitch_m[1]
        input_gram = _gram_summary(operator.input_modes, area)
        output_gram = _gram_summary(operator.output_modes, area)
        phase_zero = bool(torch.equal(encoder_phase, torch.zeros_like(encoder_phase)) and torch.equal(decoder_phase, torch.zeros_like(decoder_phase)))
        generator_trainable = [name for name, parameter in bundle.generator.named_parameters() if parameter.requires_grad]
        optical_trainable = [name for name, parameter in bundle.system.named_parameters() if parameter.requires_grad]
        equivalence = _equivalence_probe(bundle, device)
        if not phase_zero or not generator_trainable or optical_trainable:
            raise ValueError("interface lifecycle requires generator-only seeded zero-phase optics")
        settings = config.conditional_generation
        if settings is None:
            raise ValueError("conditional generation settings are required")
        if primary is None or secondary is None or lineage.get("qualification") is None:
            raise ValueError("evaluator qualification was not loaded")
        payload = {
            "schema": "fiber_interface_validation.v1",
            "gate": "valid",
            "git_commit": _git(),
            "status": "synthetic diagnostic-only",
            "formal_promotion": False,
            "training_authorized": False,
            "upstream_validation_sha256": file_sha256(root / "fiber_image_validation.json"),
            "v3_closeout_sha256": bound["v3_closeout_sha256"],
            "attribution_measurement_sha256": bound["attribution_measurement_sha256"],
            "lp01_fixture_sha256": bound["lp01_fixture_sha256"],
            "research_initialization_checkpoint_sha256": bound["research_initialization_checkpoint_sha256"],
            "source_training_evidence_sha256": bound["source_training_evidence_sha256"],
            "config_hash": upstream["config_hash"],
            "dataset_split_manifest_hash": upstream["dataset_split_manifest_hash"],
            "primary_evaluator_sha256": upstream["primary_evaluator_sha256"],
            "secondary_evaluator_sha256": upstream["secondary_evaluator_sha256"],
            "pool_hash": upstream["pool_hash"],
            "mode_bank_spec": asdict(FiberArraySpec()),
            "mode_bank_spec_sha256": _mode_bank_spec_hash(),
            "panel_seeds": list(PANELS),
            "evaluator_qualification": upstream["qualification"],
            "panels": list(PANELS),
            "variants": list(VARIANTS),
            "sample_count_per_panel": 500,
            "full_state_audit": {
                "checkpoint_load_scope": "generator_only",
                "encoder_phase_zero": bool(torch.equal(encoder_phase, torch.zeros_like(encoder_phase))),
                "decoder_phase_zero": bool(torch.equal(decoder_phase, torch.zeros_like(decoder_phase))),
                "encoder_phase_max_abs": float(encoder_phase.abs().max()),
                "decoder_phase_max_abs": float(decoder_phase.abs().max()),
                "encoder_phase_norm": float(encoder_phase.norm()),
                "decoder_phase_norm": float(decoder_phase.norm()),
                "generator_trainable_parameters": generator_trainable,
                "optical_trainable_parameters": optical_trainable,
            },
            "mode_bank": {"input": input_gram, "output": output_gram, "spec": asdict(operator.spec)},
            "equivalence": equivalence,
            "no_optimizer": True,
            "optimizer_created": False,
            "source_checkpoint_state_keys": source_state_keys,
            "loaded_generator_state_keys": loaded_generator_keys,
            "ignored_state_keys": ignored_state_keys,
            "source_phase": {
                "encoder_norm": float(source_encoder_phase.norm()),
                "encoder_max_abs": float(source_encoder_phase.abs().max()),
                "decoder_norm": float(source_decoder_phase.norm()),
                "decoder_max_abs": float(source_decoder_phase.abs().max()),
            },
            "plane": {"shape": list(operator.input_plane.shape), "pixel_pitch_m": list(operator.input_plane.pixel_pitch_m), "fov_m": list(operator.input_plane.fov_m), "wavelength_m": operator.input_plane.wavelength_m, "plane_id": operator.input_plane.plane_id},
            "harness_gates": {
                "input_mode_rank": input_gram["rank"] == 100,
                "output_mode_rank": output_gram["rank"] == 100,
                "input_gram_identity": input_gram["gram_identity_max_abs"] <= 1.0e-5,
                "output_gram_identity": output_gram["gram_identity_max_abs"] <= 1.0e-5,
                "mode_bank_finite": input_gram["finite"] and output_gram["finite"],
                "phase_state": phase_zero,
                "no_optical_trainable_parameters": not optical_trainable,
                "r0_s0_equivalence": equivalence["r0_vs_s0_complex_l2"] <= 1.0e-5,
                "r100_i0_equivalence": equivalence["r100_vs_i0_complex_l2"] <= 1.0e-5,
                "no_optimizer": True,
                "evaluator_qualification": True,
                "plane_provenance": (
                    operator.input_plane.shape == (100, 100)
                    and all(abs(actual - expected) <= 1.0e-15 for actual, expected in zip(operator.input_plane.pixel_pitch_m, (2.0e-6, 2.0e-6), strict=True))
                    and all(abs(actual - expected) <= 1.0e-15 for actual, expected in zip(operator.input_plane.fov_m, (200.0e-6, 200.0e-6), strict=True))
                    and abs(operator.input_plane.wavelength_m - 532.0e-9) <= 1.0e-15
                    and operator.input_plane.plane_id == "encoder_input"
                ),
                "device_cuda": device.type == "cuda",
                "fixed_exposure_positive": settings.fixed_exposure_scale > 0.0,
            },
        }
        payload["gate"] = "valid" if all(payload["harness_gates"].values()) else "invalid_harness"
        _write(output, payload)
        return 0 if payload["gate"] == "valid" else 2
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        _write(output, {"schema": "fiber_interface_validation.v1", "gate": "invalid_harness", "reason": str(exc), "formal_promotion": False, "training_authorized": False})
        return 2


def run(args: Any) -> int:
    validation_path = _require_file(args, "validation_evidence")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("schema") != "fiber_interface_validation.v1" or validation.get("gate") != "valid":
        raise ValueError("invalid interface attribution validation")
    if _dirty() or validation.get("git_commit") != _git():
        raise ValueError("interface validation is stale or worktree is dirty")
    _assert_validation_lineage(args, validation)
    context = _context(args, validation_path.parent)
    root = _root(args.output, require_new=True)
    all_metrics: dict[str, dict[str, Any]] = {variant: {} for variant in VARIANTS}
    embeddings: dict[str, dict[str, dict[int, torch.Tensor]]] = {variant: {stage: {} for stage in STAGES} for variant in VARIANTS}
    labels_by_panel: dict[int, torch.Tensor] = {}
    codes_by_panel: dict[int, torch.Tensor] = {}
    label_hashes: dict[str, dict[str, str]] = {variant: {} for variant in VARIANTS}
    code_hashes: dict[str, dict[str, str]] = {variant: {} for variant in VARIANTS}
    correctness: dict[str, dict[int, torch.Tensor]] = {variant: {} for variant in VARIANTS}
    for panel in PANELS:
        identity_stages: dict[str, torch.Tensor] | None = None
        s0_stages: dict[str, torch.Tensor] | None = None
        for variant in VARIANTS:
            metric, stage_tensors, stage_embeddings, labels, values = _run_variant(context, variant, panel, root)
            refs = _save_variant_artifacts(root, variant, panel, values)
            metric["artifacts"] = refs
            all_metrics[variant][str(panel)] = metric
            labels_by_panel[panel] = labels
            codes_by_panel[panel] = values["codes"]
            label_hashes[variant][str(panel)] = _tensor_hash(values["labels"])
            code_hashes[variant][str(panel)] = _tensor_hash(values["codes"])
            correctness[variant][panel] = values["secondary_correctness"].float()
            for stage, embedding in stage_embeddings.items():
                embeddings[variant][stage][panel] = embedding
            if identity_stages is None:
                identity_stages = stage_tensors
            else:
                metric["stagewise_complex_l2_vs_i0"] = _stage_l2(stage_tensors, identity_stages)
            if variant == "S0":
                s0_stages = stage_tensors
            elif variant == "L0":
                if s0_stages is None:
                    raise ValueError("S0 reference was not measured before L0")
                metric["stagewise_complex_l2_vs_s0"] = _stage_l2(stage_tensors, s0_stages)
            all_metrics[variant][str(panel)] = metric
        if identity_stages is None:
            raise ValueError("identity reference was not measured")
    nearest = _cross_panel_nearest(embeddings, labels_by_panel)
    for variant in VARIANTS:
        for panel in PANELS:
            all_metrics[variant][str(panel)]["nearest_centroid"] = nearest[variant]
    bootstrap = _paired_bootstrap(correctness, CANDIDATE_VARIANTS)
    energy_oracle_bootstrap = _paired_bootstrap(correctness, ("S0-energy-oracle",))
    r100_equivalence = {
        f"{panel}:{stage}": all_metrics["R100"][str(panel)][
            "stagewise_complex_l2_vs_i0"
        ][stage]
        for panel in PANELS
        for stage in STAGES[:-1]
    }
    r100_equivalence.update(
        {
            f"{panel}:fixed_cmos": _relative_l2(
                embeddings["R100"]["fixed_cmos"][panel],
                embeddings["I0"]["fixed_cmos"][panel],
            )
            for panel in PANELS
        }
    )
    physical_rows = [
        all_metrics[variant][str(panel)]
        for variant in PHYSICAL_VARIANTS
        for panel in PANELS
    ]
    l0_equivalence = {
        f"{panel}:{stage}": all_metrics["L0"][str(panel)][
            "stagewise_complex_l2_vs_s0"
        ][stage]
        for panel in PANELS
        for stage in STAGES[:-1]
    }
    l0_equivalence.update(
        {
            f"{panel}:fixed_cmos": _relative_l2(
                embeddings["L0"]["fixed_cmos"][panel],
                embeddings["S0"]["fixed_cmos"][panel],
            )
            for panel in PANELS
        }
    )
    pairing_valid = (
        all(
            len({label_hashes[variant][str(panel)] for variant in VARIANTS}) == 1
            for panel in PANELS
        )
        and all(
            len({code_hashes[variant][str(panel)] for variant in VARIANTS}) == 1
            for panel in PANELS
        )
    )
    harness_gates = {
        "pairing_labels": pairing_valid,
        "pairing_codes": pairing_valid,
        "energy_closure": all(
            row.get("energy_closure_error_max", float("inf")) <= 1.0e-5
            for row in physical_rows
        ),
        "passivity": all(row.get("passive") is True for row in physical_rows),
        "finite_nonnegative": all(
            row.get("finite") is True
            and row.get("nonnegative") is True
            and row.get("stage_finite") is True
            for row in physical_rows
        ),
        "r100_i0_equivalence": max(r100_equivalence.values()) <= 1.0e-5,
        "l0_s0_equivalence": max(l0_equivalence.values()) <= 1.0e-5,
    }
    jacobian = _jacobian_summary(context, codes_by_panel)
    harness_gates["jacobian"] = jacobian["finite"]
    shuffled_scores = [
        nearest[variant][stage][key]
        for variant in VARIANTS
        for stage in STAGES
        for key in ("shuffled_raw", "shuffled_unit")
    ]
    harness_gates["shuffled_label"] = max(shuffled_scores) <= 0.20
    storage_root = root.parent if root.name == "measurement" else root
    storage = sum(path.stat().st_size for path in storage_root.rglob("*") if path.is_file())
    harness_gates["storage"] = storage <= STORAGE_LIMIT_BYTES
    payload = {
        "schema": "fiber_interface_measurement.v1",
        "gate": "valid" if all(harness_gates.values()) else "invalid_harness",
        "git_commit": _git(),
        "validation_sha256": file_sha256(validation_path),
        "v3_closeout_sha256": validation["v3_closeout_sha256"],
        "attribution_measurement_sha256": validation["attribution_measurement_sha256"],
        "lp01_fixture_sha256": validation["lp01_fixture_sha256"],
        "research_initialization_checkpoint_sha256": validation[
            "research_initialization_checkpoint_sha256"
        ],
        "source_training_evidence_sha256": validation["source_training_evidence_sha256"],
        "config_hash": validation["config_hash"],
        "dataset_split_manifest_hash": validation["dataset_split_manifest_hash"],
        "primary_evaluator_sha256": validation["primary_evaluator_sha256"],
        "secondary_evaluator_sha256": validation["secondary_evaluator_sha256"],
        "pool_hash": validation["pool_hash"],
        "mode_bank_spec_sha256": validation["mode_bank_spec_sha256"],
        "variants": list(VARIANTS),
        "panels": list(PANELS),
        "rows": all_metrics,
        "nearest_centroid": nearest,
        "shuffled_label_repetitions": SHUFFLED_LABEL_REPETITIONS,
        "bootstrap": bootstrap,
        "energy_oracle_bootstrap": energy_oracle_bootstrap,
        "pairing": {"labels": label_hashes, "codes": code_hashes},
        "equivalence": {
            "r100_vs_i0_complex_l2": r100_equivalence,
            "l0_vs_s0_complex_l2": l0_equivalence,
        },
        "jacobian": jacobian,
        "harness_gates": harness_gates,
        "energy_oracle_policy": {
            "variant": "S0-energy-oracle",
            "nonphysical_upper_bound": True,
            "excluded_from_passivity": True,
            "training_authorized": False,
        },
        "storage_bytes": storage,
        "storage_limit_bytes": STORAGE_LIMIT_BYTES,
        "status": "synthetic diagnostic-only",
        "formal_promotion": False,
        "training_authorized": False,
    }
    _write(root / "measurement.json", payload)
    return 0 if payload["gate"] == "valid" else 2


def _paired_bootstrap(
    correctness: dict[str, dict[int, torch.Tensor]],
    variants: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(90210)
    output: dict[str, dict[str, Any]] = {}
    for variant in variants:
        deltas = [correctness[variant][panel] - correctness["S0"][panel] for panel in PANELS]

        def ci(delta: torch.Tensor) -> list[float]:
            samples = []
            for _ in range(10000):
                index = torch.randint(delta.numel(), (delta.numel(),), generator=generator)
                samples.append(delta[index].mean())
            values = torch.stack(samples)
            return [float(torch.quantile(values, 0.025)), float(torch.quantile(values, 0.975))]

        samples = []
        for _ in range(10000):
            panel_means = []
            for delta in deltas:
                index = torch.randint(delta.numel(), (delta.numel(),), generator=generator)
                panel_means.append(delta[index].mean())
            samples.append(torch.stack(panel_means).mean())
        values = torch.stack(samples)
        output[variant] = {
            "observed_gain": float(torch.cat(deltas).mean()),
            "ci95": [float(torch.quantile(values, 0.025)), float(torch.quantile(values, 0.975))],
            "panel_observed_gain": {
                str(panel): float(delta.mean()) for panel, delta in zip(PANELS, deltas, strict=True)
            },
            "panel_ci95": {
                str(panel): ci(delta) for panel, delta in zip(PANELS, deltas, strict=True)
            },
            "class_observed_gain": {
                str(panel): [float(value.mean()) for value in delta.reshape(10, 50)]
                for panel, delta in zip(PANELS, deltas, strict=True)
            },
            "class_ci95": {
                str(panel): [ci(value) for value in delta.reshape(10, 50)]
                for panel, delta in zip(PANELS, deltas, strict=True)
            },
            "replicates": 10000,
            "seed": 90210,
        }
    return output


def _validate_measurement_artifacts(root: Path, payload: dict[str, Any]) -> None:
    required = (
        "labels",
        "codes",
        "fixed_cmos",
        "primary_prediction",
        "secondary_prediction",
        "secondary_correctness",
        "representative_trace",
    )
    for variant in payload.get("variants", ()):
        for panel in payload.get("panels", ()):
            row = payload["rows"][variant][str(panel)]
            artifacts = row.get("artifacts")
            if not isinstance(artifacts, dict):
                raise ValueError("measurement artifact references are missing")
            for name in required:
                reference = artifacts.get(name)
                if not isinstance(reference, dict):
                    raise ValueError(f"measurement artifact reference is missing: {name}")
                path = _resolve(reference, root)
                if name == "representative_trace":
                    continue
                value = safe_torch_load(path, map_location="cpu")
                expected_tensor_hash = reference.get("tensor_sha256")
                if expected_tensor_hash != _tensor_hash(value):
                    raise ValueError(f"measurement tensor hash mismatch: {name}")
                if name in {"labels", "codes"}:
                    expected = payload["pairing"][name][variant][str(panel)]
                    if expected != _tensor_hash(value):
                        raise ValueError(f"measurement pairing hash mismatch: {name}")
            if variant in PHYSICAL_VARIANTS or variant == "S0-energy-oracle":
                for name in ("port_input_coefficients", "port_output_coefficients"):
                    reference = artifacts.get(name)
                    if not isinstance(reference, dict):
                        raise ValueError(f"port artifact reference is missing: {name}")
                    path = _resolve(reference, root)
                    value = safe_torch_load(path, map_location="cpu")
                    if reference.get("tensor_sha256") != _tensor_hash(value):
                        raise ValueError(f"port tensor hash mismatch: {name}")


def _load_measurement(path: Path, *, require_current: bool) -> tuple[Path, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "fiber_interface_measurement.v1" or payload.get("gate") != "valid":
        raise ValueError("invalid interface attribution measurement")
    _validate_measurement_artifacts(path.parent, payload)
    if require_current and (_dirty() or payload.get("git_commit") != _git()):
        raise ValueError("interface measurement is stale or worktree is dirty")
    return path.parent, payload


def _classification(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    s0 = [rows["S0"][str(panel)] for panel in PANELS]
    gains = payload["bootstrap"]
    admission_details = {}
    for variant, gain in gains.items():
        panel_gains = [
            rows[variant][str(panel)]["secondary_accuracy"]
            - rows["S0"][str(panel)]["secondary_accuracy"]
            for panel in PANELS
        ]
        recovered_panels = [
            panel
            for panel in PANELS
            if rows[variant][str(panel)]["secondary_min_recall"]
            > rows["S0"][str(panel)]["secondary_min_recall"]
            and rows[variant][str(panel)]["secondary_min_recall"] > 0.0
        ]
        admission_details[variant] = {
            "panel_gains": panel_gains,
            "same_direction_panel_count": sum(value > 0.0 for value in panel_gains),
            "minimum_recall_recovered_panels": recovered_panels,
            "admitted_gain_gate": gain["observed_gain"] >= 0.015,
            "admitted_ci_gate": gain["ci95"][0] > 0.0,
        }
    admitted = any(
        detail["admitted_gain_gate"]
        and detail["admitted_ci_gate"]
        and detail["same_direction_panel_count"] >= 2
        and len(detail["minimum_recall_recovered_panels"]) >= 2
        for detail in admission_details.values()
    )
    s0_nearest = payload["nearest_centroid"]["S0"]
    retained = all(s0_nearest[stage]["raw"] >= 0.90 for stage in ("ols_output", "camera", "fixed_cmos"))
    collapsed = any(sum(row["secondary_prediction_histogram"]) > 0 and max(row["secondary_prediction_histogram"]) / sum(row["secondary_prediction_histogram"]) > 0.80 for row in s0)
    if admitted:
        conclusion = "modal_interface_candidate_admitted"
    elif retained and (
        all(row["secondary_accuracy"] <= 0.20 for row in s0) or collapsed
    ):
        conclusion = "class_encoding_retained_semantic_manifold_mismatch"
    else:
        jacobian_points = payload.get("jacobian", {}).get("points", [])
        port_limited_points = [
            point
            for point in jacobian_points
            if point.get("port_rank_1e-3", 8) < 8
            and point.get("cmos_rank_1e-3", 8) >= point.get("port_rank_1e-3", 8) - 1
        ]
        conclusion = (
            "port_observability_limited"
            if len(port_limited_points) > len(jacobian_points) / 2
            and jacobian_points
            else "interface_inconclusive"
        )
    oracle_gain = payload.get("energy_oracle_bootstrap", {}).get("S0-energy-oracle")
    oracle_only = (
        not admitted
        and oracle_gain is not None
        and oracle_gain["observed_gain"] >= 0.015
        and oracle_gain["ci95"][0] > 0.0
    )
    return {
        "conclusion": conclusion,
        "physical_candidate_gains": gains,
        "admission_details": admission_details,
        "s0_secondary_rows": s0,
        "s0_nearest_centroid": s0_nearest,
        "energy_oracle_only": oracle_only,
        "energy_oracle_row": rows["S0-energy-oracle"],
    }


def report(args: Any) -> int:
    evidence = _require_file(args, "measurement_evidence")
    _, payload = _load_measurement(evidence, require_current=False)
    output = _root(args.output, require_new=True)
    classification = _classification(payload)
    with (output / "interface_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "panel", "primary_accuracy", "secondary_accuracy", "secondary_min_recall", "camera_fraction", "minimum_diversity", "coverage_ratio", "projection_fraction_mean", "residual_fraction_mean"])
        for variant in VARIANTS:
            for panel in PANELS:
                row = payload["rows"][variant][str(panel)]
                writer.writerow([variant, panel, row.get("primary_accuracy"), row.get("secondary_accuracy"), row.get("secondary_min_recall"), row.get("camera_fraction"), row.get("minimum_diversity"), row.get("coverage_ratio"), row.get("projection_fraction_mean"), row.get("residual_fraction_mean")])
    _write(output / "classification.json", classification)
    presentation = {
        "schema": "fiber_interface_presentation.v1",
        "gate": "valid",
        "measurement_sha256": file_sha256(evidence),
        "source_measurement_git_commit": payload["git_commit"],
        "report_git_commit": _git(),
        "classification": classification,
        "status": "synthetic diagnostic-only",
        "formal_promotion": False,
        "training_authorized": False,
    }
    _write(output / "presentation.json", presentation)
    return 0


def closeout(args: Any) -> int:
    measurement = _require_file(args, "measurement_evidence")
    presentation = _require_file(args, "presentation_evidence")
    _, payload = _load_measurement(measurement, require_current=True)
    presentation_payload = json.loads(presentation.read_text(encoding="utf-8"))
    if (
        presentation_payload.get("schema") != "fiber_interface_presentation.v1"
        or presentation_payload.get("measurement_sha256") != file_sha256(measurement)
        or presentation_payload.get("source_measurement_git_commit") != payload.get("git_commit")
    ):
        raise ValueError("presentation is not bound to the interface measurement")
    classification = _classification(payload)
    output = Path(args.output)
    next_actions = {
        "modal_interface_candidate_admitted": "register_modal_overlap_regularizer_candidate_only",
        "class_encoding_retained_semantic_manifold_mismatch": "register_zero_phase_decoder_controllability_attribution",
        "port_observability_limited": "register_mode_bank_density_geometry_no_update",
        "interface_inconclusive": "close_current_fiber_route_no_e1",
    }
    next_action = (
        "loss_exposure_budget_only"
        if classification["energy_oracle_only"]
        else next_actions[classification["conclusion"]]
    )
    storage_root = measurement.parent.parent if measurement.parent.name == "measurement" else measurement.parent
    storage_bytes = sum(path.stat().st_size for path in storage_root.rglob("*") if path.is_file())
    if storage_bytes > STORAGE_LIMIT_BYTES:
        raise ValueError("interface attribution storage gate failed at closeout")
    _write(output, {
        "schema": "fiber_interface_closeout.v1",
        "gate": "valid",
        "conclusion": classification["conclusion"],
        "measurement_sha256": file_sha256(measurement),
        "presentation_sha256": file_sha256(presentation),
        "validation_sha256": payload["validation_sha256"],
        "v3_closeout_sha256": payload["v3_closeout_sha256"],
        "attribution_measurement_sha256": payload["attribution_measurement_sha256"],
        "lp01_fixture_sha256": payload["lp01_fixture_sha256"],
        "config_hash": payload["config_hash"],
        "dataset_split_manifest_hash": payload["dataset_split_manifest_hash"],
        "primary_evaluator_sha256": payload["primary_evaluator_sha256"],
        "secondary_evaluator_sha256": payload["secondary_evaluator_sha256"],
        "mode_bank_spec_sha256": payload["mode_bank_spec_sha256"],
        "panel_seeds": payload["panels"],
        "storage_bytes": storage_bytes,
        "storage_limit_bytes": STORAGE_LIMIT_BYTES,
        "formal_promotion": False,
        "training_authorized": False,
        "synthetic_diagnostic_only": True,
        "next_action": next_action,
    })
    return 0


def dispatch(args: Any) -> int:
    return {"validate": validate, "run": run, "report": report, "closeout": closeout}[args.research_action](args)
