"""No-training single-mode modal-density upper bounds.

P196 and P400 are explicitly orthogonalized information upper bounds.  They
are useful for separating projection capacity from a readout limitation, but
the construction is not a claim about a realizable multicore fibre.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ldm_project.v2.contracts import PlaneSpec
from ldm_project.v2.evidence import atomic_write_json, file_sha256, tensor_sha256
from ldm_project.v2.optics.fiber_operator import FiberArraySpec, _mode_bank, lp01_eigenpair, lp01_radial_profile
from ldm_project.v2.application.research.capacity.wae_models import FIXED_EXPOSURE_SCALE, render_view
from ldm_project.v2.application.research.capacity.electronic_o0_wae import load_decoder_checkpoint, panel_codes

SCHEMA_PREFIX = "smf_modal_density_attribution"
MODES = (100, 196, 400)
PANELS = (4242, 104729, 161803)
STORAGE_LIMIT_BYTES = 2 * 1024**3
SMF_V2_IMPLEMENTED = False


def require_smf_v2_implementation() -> None:
    """Prevent the historical measurement placeholder from issuing evidence."""
    if not SMF_V2_IMPLEMENTED:
        raise RuntimeError(
            "SMF modal-density attribution v1 is non-consumable: O0-G v2 "
            "admission and the formal OI0/ridge/exposure/evaluator contract "
            "must be implemented before this lifecycle can run"
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


def _plane_from_config(config: Any) -> PlaneSpec:
    system = config.system
    return PlaneSpec(
        "encoder_input",
        system.metasurface_shape,
        (system.metasurface_pixel_pitch_m,) * 2,
        system.wavelength_m,
        0.0,
        "encoder",
    )


def _normalize_modes(modes: torch.Tensor, area: float) -> torch.Tensor:
    norms = (modes.abs().square().flatten(1).sum(1) * area).clamp_min(1.0e-30).sqrt()
    return modes / norms[:, None, None]


def _candidate_atoms(plane: PlaneSpec, spec: FiberArraySpec) -> torch.Tensor:
    h, w = plane.shape
    y = (torch.arange(h, dtype=torch.float64) + 0.5) * plane.pixel_pitch_m[
        0
    ] - plane.fov_m[0] / 2.0
    x = (torch.arange(w, dtype=torch.float64) + 0.5) * plane.pixel_pitch_m[
        1
    ] - plane.fov_m[1] / 2.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    _, _, rho_e = lp01_eigenpair(spec.v_number)
    radius = spec.waist_m / rho_e
    centers = torch.linspace(
        -plane.fov_m[0] * spec.fill / 2.0,
        plane.fov_m[0] * spec.fill / 2.0,
        20,
        dtype=torch.float64,
    )
    atoms = []
    for yi, cy in enumerate(centers):
        for xi, cx in enumerate(centers):
            profile = lp01_radial_profile(
                torch.sqrt((yy - cy).square() + (xx - cx).square()) / radius,
                spec.v_number,
            ).to(torch.float64)
            atoms.append((profile, yi, xi))
    return torch.stack([value[0] for value in atoms]), torch.tensor(
        [[value[1], value[2]] for value in atoms], dtype=torch.int64
    )


def build_nested_modal_banks(
    plane: PlaneSpec, *, spec: FiberArraySpec = FiberArraySpec()
) -> dict[int, torch.Tensor]:
    """Build P100 ⊂ P196 ⊂ P400 by deterministic float64 MGS."""
    area = plane.pixel_pitch_m[0] * plane.pixel_pitch_m[1]
    base = _mode_bank(plane, spec, offset_m=(0.0, 0.0), lowdin=False).real.to(
        torch.float64
    )
    base = _normalize_modes(base, area)
    atoms, coordinates = _candidate_atoms(plane, spec)
    selected = [base[index].clone() for index in range(100)]
    residuals = []
    for index in range(atoms.shape[0]):
        value = atoms[index].clone()
        for mode in selected:
            value = value - (mode.mul(value).sum() * area) * mode
        residuals.append(value)
    order: list[int] = []
    work = set(range(len(residuals)))
    for _ in range(300):
        choice = max(
            work,
            key=lambda index: (
                float(residuals[index].square().sum()),
                -int(coordinates[index, 0]),
                -int(coordinates[index, 1]),
            ),
        )
        work.remove(choice)
        norm = (residuals[choice].square().sum() * area).sqrt().clamp_min(1.0e-30)
        mode = residuals[choice] / norm
        selected.append(mode)
        for index in work:
            residuals[index] = (
                residuals[index]
                - (mode.mul(residuals[index]).sum() * area) * mode
            )
        order.append(choice)
    return {count: torch.stack(selected[:count]).to(torch.complex64) for count in MODES}


def gram_summary(bank: torch.Tensor, area: float) -> dict[str, Any]:
    gram = bank.flatten(1).conj() @ bank.flatten(1).T * area
    singular = torch.linalg.svdvals(gram.real)
    return {
        "shape": list(bank.shape),
        "rank": int(torch.linalg.matrix_rank(gram.real, tol=1.0e-5).item()),
        "condition": float(torch.linalg.cond(gram.real)),
        "max_identity_error": float(
            (gram - torch.eye(bank.shape[0], dtype=gram.dtype)).abs().max()
        ),
        "singular_values": [float(value) for value in singular],
    }


def project_reemit(
    field: torch.Tensor, bank: torch.Tensor, area: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if field.shape[-2:] != bank.shape[-2:]:
        raise ValueError("field and modal bank spatial shapes differ")
    coefficients = torch.einsum("mhw,bhw->bm", bank.conj(), field) * area
    output = torch.einsum("bm,mhw->bhw", coefficients, bank)
    return coefficients, output


def choose_ridge_lambda(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    candidates: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
) -> tuple[float, dict[str, float]]:
    if features.shape[0] != 5000 or targets.shape[0] != 5000:
        raise ValueError("ridge design requires the registered 5000 balanced samples")
    fit_x, test_x, fit_y, test_y = (
        features[:4000],
        features[4000:],
        targets[:4000],
        targets[4000:],
    )
    gram = fit_x.T @ fit_x
    values = {}
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    for value in candidates:
        weights = torch.linalg.solve(
            gram + value * torch.trace(gram) / gram.shape[0] * identity, fit_x.T @ fit_y
        )
        values[str(value)] = float((test_x @ weights - test_y).square().mean())
    selected = min(candidates, key=lambda value: values[str(value)])
    return selected, values


def _lineage(args: argparse.Namespace) -> dict[str, Any]:
    paths = [
        args.config,
        args.o0_g_decoder,
        args.o0_i_closeout,
        args.o0_g_closeout,
        args.fiber_closeout,
        args.lp01_fixture,
        args.primary_evaluator_checkpoint,
        args.secondary_evaluator_checkpoint,
        args.primary_qualification_evidence,
        args.secondary_qualification_evidence,
    ]
    missing = [str(value) for value in paths if not Path(value).is_file()]
    if missing:
        raise FileNotFoundError("SMF lineage input missing: " + ", ".join(missing))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("formal SMF attribution requires CUDA")
    return {
        "git": _git(),
        "config_sha256": file_sha256(args.config),
        "o0_g_decoder_sha256": file_sha256(args.o0_g_decoder),
        "o0_i_closeout_sha256": file_sha256(args.o0_i_closeout),
        "o0_g_closeout_sha256": file_sha256(args.o0_g_closeout),
        "fiber_closeout_sha256": file_sha256(args.fiber_closeout),
        "lp01_fixture_sha256": file_sha256(args.lp01_fixture),
        "primary_evaluator_sha256": file_sha256(args.primary_evaluator_checkpoint),
        "secondary_evaluator_sha256": file_sha256(args.secondary_evaluator_checkpoint),
        "device": args.device,
    }


def dispatch(args: argparse.Namespace) -> int:
    require_smf_v2_implementation()
    if args.research_action == "validate":
        lineage = _lineage(args)
        from ldm_project.v2.configuration.loading import load_config

        config = load_config(args.config)
        if not lineage["git"]["clean"]:
            raise RuntimeError("SMF validation requires a clean worktree")
        plane = _plane_from_config(config)
        spec = FiberArraySpec()
        banks = build_nested_modal_banks(plane, spec=spec)
        area = plane.pixel_pitch_m[0] * plane.pixel_pitch_m[1]
        grams = {str(count): gram_summary(bank, area) for count, bank in banks.items()}
        nested = {
            "100_in_196_error": float((banks[100] - banks[196][:100]).abs().max()),
            "196_in_400_error": float((banks[196] - banks[400][:196]).abs().max()),
        }
        payload = {
            "schema": f"{SCHEMA_PREFIX}_validation.v1",
            "status": "valid",
            "lineage": lineage,
            "mode_banks": grams,
            "nested_basis_error": nested,
            "modes": list(MODES),
            "physical_parameter_status": "synthetic_diagnostic_only",
            "formal_promotion": False,
            "training_authorized": False,
            "synthetic_diagnostic_only": True,
        }
        _write(args.output, payload)
        return 0
    if args.research_action == "run":
        validation = json.loads(
            Path(args.validation_evidence).read_text(encoding="utf-8")
        )
        if validation.get("status") != "valid":
            raise ValueError("SMF run requires valid validation evidence")
        from ldm_project.v2.configuration.loading import load_config

        config = load_config(args.config)
        plane = _plane_from_config(config)
        spec = FiberArraySpec()
        banks = build_nested_modal_banks(plane, spec=spec)
        area = plane.pixel_pitch_m[0] * plane.pixel_pitch_m[1]
        decoder = load_decoder_checkpoint(
            args.o0_g_decoder, candidate="O0-G", device=torch.device(args.device)
        )
        rows: dict[str, Any] = {}
        for panel in PANELS:
            codes, labels = panel_codes(panel, device=torch.device(args.device))
            with torch.no_grad():
                commands = decoder(codes, labels)
                ideal = render_view(commands, "O0-G")
                field = (
                    F.interpolate(commands, size=(100, 100), mode="area")
                    .squeeze(1)
                    .to(torch.complex64)
                )
            panel_row: dict[str, Any] = {
                "labels_hash": tensor_sha256(labels.cpu()),
                "codes_hash": tensor_sha256(codes.cpu()),
                "command_hash": tensor_sha256(commands.cpu()),
                "branches": {"E0": {"view_hash": tensor_sha256(ideal.cpu())}},
            }
            for count, bank in banks.items():
                coefficients, output = project_reemit(
                    field, bank.to(field.device), area
                )
                view = F.adaptive_avg_pool2d(
                    output.abs().square() / FIXED_EXPOSURE_SCALE, (28, 28)
                ).clamp(0, 1)
                panel_row["branches"][f"P{count}"] = {
                    "view_hash": tensor_sha256(view.cpu()),
                    "coefficients_hash": tensor_sha256(coefficients.cpu()),
                    "retention": float(
                        (
                            output.abs().square().sum((-2, -1))
                            / field.abs().square().sum((-2, -1)).clamp_min(1e-30)
                        ).mean()
                    ),
                    "energy_closure": float(
                        (
                            (
                                field.abs().square().sum((-2, -1))
                                - output.abs().square().sum((-2, -1))
                            ).abs()
                            / field.abs().square().sum((-2, -1)).clamp_min(1e-30)
                        ).max()
                    ),
                }
            rows[str(panel)] = panel_row
        payload = {
            "schema": f"{SCHEMA_PREFIX}_measurement.v1",
            "status": "valid",
            "lineage": validation.get("lineage"),
            "panels": rows,
            "gram": validation["mode_banks"],
            "ridge": {"evaluator_calls": 0, "selection": "not_run_in_measurement"},
            "physical_parameter_status": "synthetic_diagnostic_only",
            "formal_promotion": False,
            "training_authorized": False,
            "synthetic_diagnostic_only": True,
        }
        _write(args.output, payload)
        return 0
    measurement = json.loads(
        Path(args.measurement_evidence).read_text(encoding="utf-8")
    )
    if measurement.get("status") != "valid":
        raise ValueError("SMF report/closeout requires valid measurement")
    if args.research_action == "report":
        payload = {
            "schema": f"{SCHEMA_PREFIX}_presentation.v1",
            "status": "valid",
            "measurement_sha256": file_sha256(args.measurement_evidence),
            "classification": "smf_upper_bound_inconclusive",
            "next_action": "hold_no_training_request_new_hypothesis",
            "formal_promotion": False,
            "training_authorized": False,
            "synthetic_diagnostic_only": True,
        }
        _write(args.output, payload)
        return 0
    presentation = json.loads(
        Path(args.presentation_evidence).read_text(encoding="utf-8")
    )
    payload = {
        "schema": f"{SCHEMA_PREFIX}_closeout.v1",
        "status": "valid",
        "classification": presentation.get(
            "classification", "smf_upper_bound_inconclusive"
        ),
        "next_action": presentation.get(
            "next_action", "hold_no_training_request_new_hypothesis"
        ),
        "measurement_sha256": file_sha256(args.measurement_evidence),
        "presentation_sha256": file_sha256(args.presentation_evidence),
        "formal_promotion": False,
        "training_authorized": False,
        "synthetic_diagnostic_only": True,
    }
    _write(args.output, payload)
    return 0
