"""HW207 MNIST4 C3.7 shared numerical library: fixed focus cores, forward, losses.

Scope.  This module reuses the frozen C3.4 scalar Fresnel Model-B forward, the
C3.3 binary DMD input proxy and the C3.3 derived fixed 2x2 CMOS ROI.  It adds:

* four frozen target focus cores ``C_0..C_3`` strictly inside ``ROI_0..ROI_3``;
* differentiable core / sidelobe / spread / throughput terms;
* the single shared evaluation path used by training selection, P0 diagnosis
  and the P4 fabrication comparison.

Nothing here is a calibrated HW207 bench, a printer model or a materials
measurement.  All geometry is the historical tape estimate recorded in
``configs/research/lightgen/hw207_mnist4_c34_v1.json``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from scripts.research.lightgen import execute_hw207_mnist4_c34_v1 as c34_runner
from scripts.research.lightgen import hw207_mnist4_c33_phase_v1 as c33
from scripts.research.lightgen import hw207_mnist4_c34_v1 as c34
from scripts.research.lightgen import hw207_mnist4_c34_process_aware_v1 as c34_process


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/hw207_mnist4_c37_v1.json"
IDENTITY = "HW207_MNIST4_C37_FOCUS_OPTIMISATION"

GRID = c34.GRID
CMOS = int(c34.CMOS_SHAPE[0])
CMOS_PITCH_UM = float(c34.CMOS_PITCH_UM)
FRAME_CENTER_PX = (CMOS - 1) / 2.0
ROI_SIDE_PX = 103
EPS = 1e-12


# --------------------------------------------------------------------------- #
# configuration and small IO helpers
# --------------------------------------------------------------------------- #
def load_config(path: Path | str | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else DEFAULT_CONFIG
    config = json.loads(source.read_text(encoding="utf-8"))
    if config.get("identity") != IDENTITY:
        raise ValueError("unexpected C3.7 configuration identity")
    if config.get("wavelength_nm") != 532.0:
        raise ValueError("C3.7 is frozen to the 532 nm numerical scenario")
    geometry = config["geometry"]
    if geometry["phase_grid"] != [GRID, GRID] or geometry["phase_pitch_um"] != c34.PITCH_UM:
        raise ValueError("C3.7 phase grid or pitch changed")
    expected = {
        "screen_to_lens_mm": c34.MODEL_B_D1_UM / 1000.0,
        "lens_focal_length_mm": c34.MODEL_B_F_UM / 1000.0,
        "lens_to_cmos_mm": c34.MODEL_B_D2_UM / 1000.0,
        "camera_shape": [CMOS, CMOS],
        "camera_pixel_pitch_um": CMOS_PITCH_UM,
        "lens_plane_pitch_um": 4.0,
        "aperture_diameter_mm": 2.0,
    }
    for name, value in expected.items():
        if geometry.get(name) != value:
            raise ValueError(f"C3.7 geometry field {name} does not match the frozen forward model")
    return config


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_frozen_phase(path: Path) -> torch.Tensor:
    """Load a frozen 100x100 phase matrix and reject anything else."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            value = None
            for key in ("phase", "phase_rad", "effective", "state_dict"):
                candidate = state.get(key)
                if isinstance(candidate, np.ndarray):
                    value = candidate
                    break
                if isinstance(candidate, torch.Tensor):
                    value = candidate
                    break
            if value is None:
                raise ValueError(f"no phase tensor found in {path}")
        else:
            value = state
    phase = torch.as_tensor(value, dtype=torch.float32).squeeze()
    if phase.shape != (GRID, GRID):
        raise ValueError(f"frozen phase in {path} is {tuple(phase.shape)}, expected {(GRID, GRID)}")
    if not torch.isfinite(phase).all():
        raise ValueError(f"frozen phase in {path} contains non-finite values")
    return phase.detach().clone()


# --------------------------------------------------------------------------- #
# fixed ROI and frozen target cores
# --------------------------------------------------------------------------- #
def roi_bounds() -> tuple[tuple[int, int, int, int], ...]:
    """The frozen C3.3 four-quadrant ROI translated into the C3.4 CMOS frame."""
    bounds = c34.fixed_roi_bounds("cpu")
    if len(bounds) != 4:
        raise ValueError("frozen ROI count changed")
    for r0, r1, c0, c1 in bounds:
        if (r1 - r0) != ROI_SIDE_PX or (c1 - c0) != ROI_SIDE_PX:
            raise ValueError("frozen ROI side changed")
        if not (0 <= r0 < r1 <= CMOS and 0 <= c0 < c1 <= CMOS):
            raise ValueError("frozen ROI leaves the CMOS frame")
    return tuple(tuple(int(v) for v in item) for item in bounds)


def roi_center_px(bounds: Sequence[Sequence[int]]) -> np.ndarray:
    centers = np.array([[(r0 + r1 - 1) / 2.0, (c0 + c1 - 1) / 2.0] for r0, r1, c0, c1 in bounds])
    if not np.allclose(centers, np.round(centers)):
        raise ValueError("frozen ROI centres are not on integer CMOS pixels")
    return centers


def core_side_px(fraction: float) -> int:
    """Largest odd pixel side not exceeding ``fraction`` of the ROI side."""
    if not 0.05 <= fraction <= 0.95:
        raise ValueError("core fraction must be between 0.05 and 0.95")
    side = 2 * int(math.floor(fraction * ROI_SIDE_PX / 2.0)) + 1
    if side < 3 or side > ROI_SIDE_PX - 2:
        raise ValueError(f"core fraction {fraction} gives an unusable core side {side}")
    return side


def core_bounds(fraction: float) -> tuple[tuple[int, int, int, int], ...]:
    """Four cores centred on their own ROI centre and fully inside that ROI."""
    side = core_side_px(fraction)
    half = (side - 1) // 2
    result = []
    for (r0, r1, c0, c1), (cy, cx) in zip(roi_bounds(), roi_center_px(roi_bounds())):
        cy, cx = int(cy), int(cx)
        core = (cy - half, cy + half + 1, cx - half, cx + half + 1)
        if not (r0 <= core[0] < core[1] <= r1 and c0 <= core[2] < core[3] <= c1):
            raise ValueError("target core escapes its own ROI")
        result.append(core)
    return tuple(result)


def core_spec(fraction: float) -> dict[str, Any]:
    """Machine readable, frozen description of the four target cores."""
    side = core_side_px(fraction)
    bounds = core_bounds(fraction)
    centers = roi_center_px(bounds)
    rois = roi_bounds()
    side_um = side * CMOS_PITCH_UM
    records = []
    for index, ((r0, r1, c0, c1), (cy, cx)) in enumerate(zip(bounds, centers)):
        roi = rois[index]
        records.append({
            "class_index": index,
            "mnist_digit": index,
            "roi_bounds_px": [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])],
            "core_bounds_px": [int(r0), int(r1), int(c0), int(c1)],
            "core_center_px_row_col": [int(cy), int(cx)],
            "core_center_um_y_x": [
                float((cy - FRAME_CENTER_PX) * CMOS_PITCH_UM),
                float((cx - FRAME_CENTER_PX) * CMOS_PITCH_UM),
            ],
            "core_side_px": int(side),
            "core_side_um": float(side_um),
            "core_area_um2": float(side_um ** 2),
            "core_edge_margin_to_roi_px": float((ROI_SIDE_PX - side) / 2.0),
            "core_edge_margin_to_roi_um": float((ROI_SIDE_PX - side) / 2.0 * CMOS_PITCH_UM),
            "pixel_area_um2": float(CMOS_PITCH_UM ** 2),
        })
    return {
        "identity": "C37_FROZEN_TARGET_CORES",
        "fraction_of_roi_side": float(fraction),
        "core_side_px": int(side),
        "core_side_um": float(side_um),
        "roi_side_px": ROI_SIDE_PX,
        "roi_side_um": float(ROI_SIDE_PX * CMOS_PITCH_UM),
        "cmos_frame_shape": [CMOS, CMOS],
        "cmos_pixel_pitch_um": CMOS_PITCH_UM,
        "cmos_frame_side_um": float(CMOS * CMOS_PITCH_UM),
        "array_origin": "index 0 is the lowest physical y (row) and x (column); "
                        "the optical axis is the array centre (1023.5, 1023.5)",
        "centres_are_roi_geometric_centres": True,
        "cores_do_not_cross_roi_boundaries": True,
        "records": records,
    }


_MASK_CACHE: dict[tuple[float, str], torch.Tensor] = {}


def core_masks(fraction: float, device: torch.device | str = "cpu") -> torch.Tensor:
    """Stacked (4, 2048, 2048) float32 core indicator masks."""
    key = (float(fraction), str(device))
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached
    masks = torch.zeros((4, CMOS, CMOS), dtype=torch.float32, device=device)
    for index, (r0, r1, c0, c1) in enumerate(core_bounds(fraction)):
        masks[index, r0:r1, c0:c1] = 1.0
    _MASK_CACHE[key] = masks
    return masks


def roi_masks(device: torch.device | str = "cpu") -> torch.Tensor:
    masks = torch.zeros((4, CMOS, CMOS), dtype=torch.float32, device=device)
    for index, (r0, r1, c0, c1) in enumerate(roi_bounds()):
        masks[index, r0:r1, c0:c1] = 1.0
    return masks


# --------------------------------------------------------------------------- #
# context, input fields, forward
# --------------------------------------------------------------------------- #
def build_context(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C3.7 optical propagation and training require CUDA")
    images, labels, split = c34_runner._split_and_data()
    data = config["data"]
    if len(split["train"]) != data["train_count"]:
        raise ValueError("C3.7 training rows differ from the frozen configuration")
    if len(split["validation"]) != data["validation_count"]:
        raise ValueError("C3.7 validation rows differ from the frozen configuration")
    if len(split["holdout"]) != data["previously_used_holdout_count"]:
        raise ValueError("C3.7 historical holdout rows changed")
    if torch.isin(split["train"], split["validation"]).any():
        raise ValueError("C3.7 training and validation rows overlap")
    if torch.isin(split["validation"], split["holdout"]).any():
        raise ValueError("C3.7 validation and prior holdout rows overlap")
    if (labels[torch.cat((split["train"], split["validation"], split["holdout"]))] > 3).any():
        raise ValueError("C3.7 data contains labels outside 0-3")
    configuration = c33.build_configuration(52_000.0, str(device))
    configuration.base.screen.requires_grad_(False)
    geometry = config["geometry"]
    return {
        "images": images,
        "labels": labels,
        "split": split,
        "device": device,
        "input_configuration": configuration,
        "lens_pitch_um": float(geometry["lens_plane_pitch_um"]),
        "screen_to_lens_um": float(geometry["screen_to_lens_mm"]) * 1000.0,
        "lens_to_cmos_um": float(geometry["lens_to_cmos_mm"]) * 1000.0,
        "aperture_mm": float(geometry["aperture_diameter_mm"]),
        "propagation_cache": {},
        "core_fraction": float(config["core"]["frozen_fraction"]),
        "field_cache": None,
        "field_cache_metadata": {},
    }


def prepare_field_cache(context: dict[str, Any], rows: torch.Tensor, *,
                        cache_path: Path | None = None,
                        chunk_size: int = 256) -> dict[str, Any]:
    """Build or load the deterministic frozen input-plane field cache.

    Generating one input field costs two dense matrix products against the frozen
    C3.3 relay matrices, so the cache is built once with large chunks and reused
    for every epoch and every stage.
    """
    required = torch.sort(torch.unique(torch.as_tensor(rows, dtype=torch.long).cpu())).values
    if cache_path is not None and Path(cache_path).is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        cached_rows = torch.as_tensor(payload.get("rows"), dtype=torch.long)
        fields = payload.get("fields")
        if (payload.get("identity") != "C37_FIXED_BINARY_INPUT_CACHE"
                or not torch.equal(cached_rows, required)
                or not isinstance(fields, torch.Tensor)
                or fields.shape != (len(required), GRID, GRID)
                or fields.dtype != torch.complex64
                or not torch.isfinite(fields).all()):
            raise ValueError("saved C3.7 input cache does not match the requested rows")
        context["field_cache"] = (cached_rows, fields)
        context["field_cache_metadata"] = dict(payload.get("metadata") or {})
        context["field_cache_metadata"]["loaded_from"] = str(cache_path)
        return context["field_cache_metadata"]
    cache_rows, fields, metadata = c34_runner._prepare_field_cache(
        context["images"], required, context["input_configuration"], context["device"],
        chunk_size=chunk_size)
    context["field_cache"] = (cache_rows.detach().cpu(), fields.detach().cpu())
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save({"identity": "C37_FIXED_BINARY_INPUT_CACHE",
                    "rows": context["field_cache"][0],
                    "fields": context["field_cache"][1],
                    "metadata": metadata}, temporary)
        temporary.replace(cache_path)
        metadata = dict(metadata)
        metadata["written_to"] = str(cache_path)
    context["field_cache_metadata"] = metadata
    return metadata


def input_fields(context: dict[str, Any], rows: torch.Tensor) -> torch.Tensor:
    """Deterministic frozen binary DMD input proxy for the requested MNIST rows."""
    device = context["device"]
    rows_cpu = torch.as_tensor(rows, dtype=torch.long).detach().cpu()
    cache = context.get("field_cache")
    if cache is not None:
        cache_rows, fields = cache
        positions = torch.searchsorted(cache_rows, rows_cpu)
        if (len(positions) == 0
                or int(positions.max()) < len(cache_rows)
                and torch.equal(cache_rows[positions.clamp_max(len(cache_rows) - 1)], rows_cpu)):
            return fields[positions].to(device)
    with torch.no_grad():
        return c34.input_field(context["images"][rows_cpu].to(device), context["input_configuration"])


def forward_batch(fields: torch.Tensor, phase: torch.Tensor, context: dict[str, Any], *,
                  return_camera: bool = True, aperture_mm: float | None = None,
                  lens_pitch_um: float | None = None,
                  screen_to_lens_um: float | None = None,
                  lens_to_cmos_um: float | None = None,
                  lens_to_cmos_chirp_dtype: torch.dtype | None = None,
                  plan_cache: dict[Any, Any] | None = None) -> dict[str, Any]:
    return c34.model_b_forward(
        fields,
        phase,
        float(context["aperture_mm"] if aperture_mm is None else aperture_mm),
        lens_pitch_um=float(context["lens_pitch_um"] if lens_pitch_um is None else lens_pitch_um),
        plan_cache=context["propagation_cache"] if plan_cache is None else plan_cache,
        return_camera=return_camera,
        screen_to_lens_um=float(context["screen_to_lens_um"] if screen_to_lens_um is None
                                else screen_to_lens_um),
        lens_to_cmos_um=float(context["lens_to_cmos_um"] if lens_to_cmos_um is None
                              else lens_to_cmos_um),
        lens_to_cmos_chirp_dtype=lens_to_cmos_chirp_dtype,
    )


# --------------------------------------------------------------------------- #
# differentiable spot terms
# --------------------------------------------------------------------------- #
def eta_phase_to_cmos(cmos_energy: torch.Tensor, input_energy: torch.Tensor) -> torch.Tensor:
    return cmos_energy / input_energy.clamp_min(EPS)


def spot_terms(result: dict[str, Any], labels: torch.Tensor, masks: torch.Tensor,
               context: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Per-sample energies used by the focus, sidelobe and spread terms.

    All energies are in ``um^2`` (intensity integral over the CMOS pixel area),
    matching the frozen C3.4 ``roi_scores`` convention.
    """
    intensity = result["camera_intensity"]
    if intensity is None:
        raise ValueError("spot terms require the camera intensity")
    pixel_area = CMOS_PITCH_UM ** 2
    scores = result["roi_scores"]
    rows = torch.arange(len(labels), device=labels.device)
    correct_roi = scores[rows, labels]
    core_energy = torch.stack(
        [(intensity * masks[index]).sum((1, 2)) * pixel_area for index in range(4)], dim=1)
    core = core_energy[rows, labels]
    total = result["detected_energy"].clamp_min(EPS)
    four_roi = scores.sum(1)
    wrong = four_roi - correct_roi
    outside = result["outside_roi_full_cmos_energy"]
    own_outside_core = (correct_roi - core).clamp_min(0.0)
    return {
        "core_energy_um2": core,
        "core_fraction": core / total,
        "correct_roi_fraction": correct_roi / total,
        "wrong_roi_fraction": wrong.clamp_min(0.0) / total,
        "outside_roi_fraction": outside.clamp_min(0.0) / total,
        "own_roi_outside_core_fraction": own_outside_core / total,
        "total_energy_um2": total,
        "four_roi_fraction": four_roi / total,
        "eta": eta_phase_to_cmos(result["detected_energy"],
                                 result["input_energy"].clamp_min(EPS)),
    }


def spread_radius_um(intensity: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Intensity weighted RMS radius about each sample's own frozen core centre.

    Uses the exact algebraic identity for a shifted second moment so that only
    four reductions over the frame are required.  Physical units are micrometres.
    """
    if intensity is None:
        raise ValueError("spread radius requires the camera intensity")
    dtype = intensity.dtype
    axis = torch.arange(CMOS, device=intensity.device, dtype=dtype) - FRAME_CENTER_PX
    row_profile = intensity.sum(2)
    column_profile = intensity.sum(1)
    total = row_profile.sum(1)
    moment_y = (row_profile * axis).sum(1)
    moment_x = (column_profile * axis).sum(1)
    radial = ((row_profile * axis.square()).sum(1) + (column_profile * axis.square()).sum(1))
    centers = roi_center_px(roi_bounds())
    dy = torch.as_tensor(centers[:, 0], device=intensity.device, dtype=dtype) - FRAME_CENTER_PX
    dx = torch.as_tensor(centers[:, 1], device=intensity.device, dtype=dtype) - FRAME_CENTER_PX
    dy = dy[labels]
    dx = dx[labels]
    second = radial - 2.0 * dy * moment_y - 2.0 * dx * moment_x + (dy.square() + dx.square()) * total
    second = second.clamp_min(0.0)
    return CMOS_PITCH_UM * torch.sqrt((second + 1e-18) / total.clamp_min(EPS))


# --------------------------------------------------------------------------- #
# joint loss
# --------------------------------------------------------------------------- #
def objective(result: dict[str, Any], labels: torch.Tensor, phase: torch.Tensor,
              context: dict[str, Any], config: dict[str, Any], *,
              groups: Sequence[str], eta_target: float, multiplier: float,
              masks: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint classification + focus + sidelobe + spread + throughput loss.

    ``groups`` selects the active families so that the ablation matrix
    E1..E4 shares one implementation.  The classification family always keeps
    the historical C3.5 form (cross entropy + margin + ROI leakage).
    """
    weights = config["training"]
    scores = result["roi_scores"]
    crop = result["comparison_crop_energy"].clamp_min(1e-20)
    logits = 20.0 * scores / crop[:, None]
    terms = spot_terms(result, labels, masks, context)
    rows = torch.arange(len(labels), device=labels.device)

    classification = F.cross_entropy(logits.float(), labels)
    correct = scores[rows, labels]
    incorrect = scores.clone()
    incorrect[rows, labels] = -float("inf")
    margin = F.relu(1.0 - (correct - incorrect.max(1).values) / crop).mean()
    leakage = (result["outside_roi_full_cmos_energy"] /
               result["detected_energy"].clamp_min(1e-20)).mean()
    loss = (classification
            + float(weights["classification_margin_weight"]) * margin
            + float(weights["roi_leakage_weight"]) * leakage)
    components: dict[str, torch.Tensor] = {
        "classification": classification.detach(),
        "margin": margin.detach(),
        "leakage": leakage.detach(),
    }

    if "focus" in groups:
        log_core = torch.log(terms["core_fraction"] + 1e-6)
        focus_mode = str(weights.get("focus_mode", "mean"))
        if focus_mode == "worst_class":
            present = [index for index in range(4) if bool((labels == index).any())]
            if not present:
                raise ValueError("worst_class focus needs at least one labelled sample")
            per_class = torch.stack([log_core[labels == index].mean() for index in present])
            # log_core is negative for every sample, so the smallest mean is the weakest
            # class; negating it makes the weakest class carry the largest penalty.
            focus = -per_class.min()
        else:
            focus = -log_core.mean()
        loss = loss + float(weights["focus_weight"]) * focus
        components["focus"] = focus.detach()
        components["core_fraction_mean"] = terms["core_fraction"].detach().mean()

    if "sidelobe" in groups:
        sidelobe = (float(weights["sidelobe_wrong_roi_weight"]) * terms["wrong_roi_fraction"]
                    + float(weights["sidelobe_outside_roi_weight"]) * terms["outside_roi_fraction"]
                    + float(weights["sidelobe_own_roi_outside_core_weight"])
                    * terms["own_roi_outside_core_fraction"]).mean()
        loss = loss + float(weights["sidelobe_weight"]) * sidelobe
        components["sidelobe"] = sidelobe.detach()

    if "spread" in groups:
        radius = spread_radius_um(result["camera_intensity"], labels)
        spread = (radius / float(config["core"]["spread_normalisation_um"])).square().mean()
        loss = loss + float(weights["spread_weight"]) * spread
        components["spread"] = spread.detach()
        components["spread_radius_um_mean"] = radius.detach().mean()

    if "throughput" in groups:
        shortfall = F.relu(float(eta_target) - terms["eta"])
        constraint = float(multiplier) * shortfall.mean() + 0.5 * float(
            config["energy"]["adaptive_lagrange_rho"]) * shortfall.square().mean()
        loss = loss + constraint
        components["constraint"] = constraint.detach()
        components["eta_shortfall_mean"] = shortfall.detach().mean()
        components["eta_mean"] = terms["eta"].detach().mean()

    # manufacturing bandwidth proxy: every trainable arm carries the same periodic
    # phase smoothness penalty so that the ablation isolates the focus terms
    periodic = (torch.sin((phase[..., 1:] - phase[..., :-1]) / 2.0).abs().mean()
                + torch.sin((phase[..., 1:, :] - phase[..., :-1, :]) / 2.0).abs().mean())
    loss = loss + float(weights["phase_smoothness_weight"]) * periodic
    components["phase_total_variation"] = periodic.detach()

    components["loss"] = loss.detach()
    return loss, components


# --------------------------------------------------------------------------- #
# frozen-parameter fabrication pipeline (phase -> height -> print -> phase)
# --------------------------------------------------------------------------- #
def processed_phase(phase: torch.Tensor, config: dict[str, Any], *, delta_n: float,
                    psf_sigma_px: float, minimum_feature_radius_px: int,
                    height_levels: int, height_noise_nm: float,
                    noise_seed: int, edge_height_bias_nm: float = 0.0) -> torch.Tensor:
    """Apply the assumed print operator in the physical height domain.

    Wrapped phase is converted to a physical height first; blur, minimum
    feature, quantisation and noise are then applied to that height.  Applying a
    Gaussian blur directly to wrapped phase is not equivalent and is never used.
    """
    assumptions = config["fabrication_assumptions"]
    height = c34.height_from_phase(phase, float(delta_n))
    noise = c34_process._noise_field(height, int(noise_seed), float(height_noise_nm))
    printed = c34_process.process_height(
        height,
        psf_sigma_px=float(psf_sigma_px),
        minimum_feature_radius_px=int(minimum_feature_radius_px),
        height_levels=int(height_levels),
        maximum_height_um=float(assumptions["maximum_height_um"]),
        noise_nm=float(height_noise_nm),
        noise_seed=int(noise_seed),
        edge_height_bias_nm=float(edge_height_bias_nm),
        noise_field=noise,
        straight_through_quantization=False,
    )
    return c34.phase_from_height(printed, float(delta_n))


def height_map_from_phase(phase: torch.Tensor, delta_n: float) -> torch.Tensor:
    return c34.height_from_phase(phase, float(delta_n))


# --------------------------------------------------------------------------- #
# per-sample statistics (no gradients)
# --------------------------------------------------------------------------- #
def row_statistics(intensity: torch.Tensor, scores: torch.Tensor, detected: torch.Tensor,
                   core_energies: torch.Tensor, label: int) -> dict[str, float]:
    """Single sample spot statistics in the frozen frame coordinates."""
    rois = roi_bounds()
    rois_np = np.array(rois, dtype=np.int64)
    pixel_area = CMOS_PITCH_UM ** 2
    image = intensity.detach().to(torch.float64)
    flat_index = int(torch.argmax(image))
    peak_row, peak_col = divmod(flat_index, CMOS)
    peak_value = float(image.reshape(-1)[flat_index])
    total = float(detected.to(torch.float64))
    roi_energy = scores.detach().to(torch.float64).cpu().numpy()
    row_profile = image.sum(1)
    column_profile = image.sum(0)
    axis = torch.arange(CMOS, dtype=torch.float64, device=image.device) - FRAME_CENTER_PX
    denominator = float(image.sum())
    centroid_row = float((row_profile * axis).sum() / denominator) + FRAME_CENTER_PX
    centroid_col = float((column_profile * axis).sum() / denominator) + FRAME_CENTER_PX

    target = rois[label]
    inside_roi = lambda r, c, box: bool(box[0] <= r < box[1] and box[2] <= c < box[3])  # noqa: E731
    peak_in_target = inside_roi(peak_row, peak_col, target)
    centroid_in_target = inside_roi(centroid_row, centroid_col, target)
    centre = roi_center_px(rois)[label]
    distance_um = float(math.hypot(peak_row - centre[0], peak_col - centre[1]) * CMOS_PITCH_UM)
    target_slice = image[target[0]:target[1], target[2]:target[3]]
    target_peak = float(target_slice.max())
    outside = image.clone()
    outside[target[0]:target[1], target[2]:target[3]] = 0.0
    outside_peak = float(outside.max())
    outside_energy = float(outside.sum()) * pixel_area
    core_energy = core_energies.detach().to(torch.float64).cpu().numpy()
    rois_side = ROI_SIDE_PX * CMOS_PITCH_UM
    return {
        "label": float(label),
        "roi_energies_um2": roi_energy.tolist(),
        "four_roi_energy_um2": float(roi_energy.sum()),
        "full_cmos_energy_um2": total,
        "target_roi_share_of_four_roi": float(roi_energy[label] / max(roi_energy.sum(), EPS)),
        "target_roi_share_of_cmos": float(roi_energy[label] / max(total, EPS)),
        "global_peak_intensity": peak_value,
        "global_peak_row": float(peak_row),
        "global_peak_col": float(peak_col),
        "global_peak_in_target_roi": float(peak_in_target),
        "centroid_row": centroid_row,
        "centroid_col": centroid_col,
        "centroid_in_target_roi": float(centroid_in_target),
        "peak_distance_to_target_centre_um": distance_um,
        "target_roi_peak_intensity": target_peak,
        "outside_target_roi_peak_intensity": outside_peak,
        "outside_target_roi_energy_fraction": float(outside_energy / max(total, EPS)),
        "core_energies_um2": core_energy.tolist(),
        "core_fraction": float(core_energy[label] / max(total, EPS)),
        "target_roi_side_um": float(rois_side),
        "peak_to_target_roi_centre_distance_fraction_of_roi": float(distance_um / rois_side),
    }


def aggregate_statistics(records: list[dict[str, float]], labels: np.ndarray,
                         predictions: np.ndarray, core_fractions: np.ndarray,
                         eta: np.ndarray) -> dict[str, Any]:
    """Class-wise and overall distributions for the P0 / acceptance metrics."""
    def distribution(values: np.ndarray) -> dict[str, float]:
        values = np.asarray(values, dtype=np.float64)
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "max": float(values.max()),
        }

    peak_in = np.array([record["global_peak_in_target_roi"] for record in records])
    centroid_in = np.array([record["centroid_in_target_roi"] for record in records])
    share_four = np.array([record["target_roi_share_of_four_roi"] for record in records])
    share_cmos = np.array([record["target_roi_share_of_cmos"] for record in records])
    outside_fraction = np.array([record["outside_target_roi_energy_fraction"] for record in records])
    distance = np.array([record["peak_distance_to_target_centre_um"] for record in records])
    per_class = {}
    for cls in range(4):
        mask = labels == cls
        if not mask.any():
            continue
        per_class[str(cls)] = {
            "count": int(mask.sum()),
            "accuracy": float((predictions[mask] == labels[mask]).mean()),
            "peak_localisation_rate": float(peak_in[mask].mean()),
            "centroid_localisation_rate": float(centroid_in[mask].mean()),
            "target_roi_share_of_four_roi_median": float(np.median(share_four[mask])),
            "target_roi_share_of_cmos_median": float(np.median(share_cmos[mask])),
            "core_fraction_median": float(np.median(core_fractions[mask])),
            "outside_target_roi_energy_fraction_mean": float(outside_fraction[mask].mean()),
            "peak_distance_to_target_centre_um_median": float(np.median(distance[mask])),
            "eta_median": float(np.median(eta[mask])),
        }
    confusion = np.zeros((4, 4), dtype=np.int64)
    for truth, prediction in zip(labels, predictions):
        confusion[int(truth), int(prediction)] += 1
    macro_f1 = float(np.mean([
        2 * confusion[c, c] / max(confusion[c].sum() + confusion[:, c].sum(), 1) for c in range(4)
    ]))
    return {
        "population": int(len(records)),
        "class_counts": {str(cls): int((labels == cls).sum()) for cls in range(4)},
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": macro_f1,
        "confusion_matrix": confusion.tolist(),
        "peak_localisation_rate": float(peak_in.mean()),
        "centroid_localisation_rate": float(centroid_in.mean()),
        "target_roi_share_of_four_roi": distribution(share_four),
        "target_roi_share_of_cmos": distribution(share_cmos),
        "outside_target_roi_energy_fraction": distribution(outside_fraction),
        "peak_distance_to_target_centre_um": distribution(distance),
        "core_fraction": distribution(core_fractions),
        "eta_phase_to_cmos": distribution(eta),
        "low_efficiency_fraction_below_half_median": float(
            (eta < 0.5 * np.median(eta)).mean()) if eta.size else 0.0,
        "per_class": per_class,
    }


# --------------------------------------------------------------------------- #
# single shared evaluation path
# --------------------------------------------------------------------------- #
def evaluate_phase(context: dict[str, Any], config: dict[str, Any], phase: torch.Tensor,
                   rows: torch.Tensor, *, tag: str, batch_size: int = 4,
                   save_maps_rows: Sequence[int] = (), dtype_statistics: str = "float64",
                   aperture_mm: float | None = None, lens_pitch_um: float | None = None,
                   screen_to_lens_um: float | None = None,
                   lens_to_cmos_um: float | None = None,
                   verbose: bool = True) -> dict[str, Any]:
    """Forward the frozen phase over ``rows`` and return metrics and per-row data."""
    if dtype_statistics not in {"float32", "float64"}:
        raise ValueError("unsupported statistics dtype")
    device = context["device"]
    fraction = float(config["core"]["frozen_fraction"])
    masks = core_masks(fraction, device)
    labels_all = context["labels"]
    phase = phase.to(device=device, dtype=torch.float32).detach()
    rows_cpu = torch.as_tensor(rows, dtype=torch.long).detach().cpu()
    labels_np = labels_all[rows_cpu].numpy().astype(np.int64)
    save_set = {int(value) for value in save_maps_rows}
    records: list[dict[str, float]] = []
    scores_out: list[np.ndarray] = []
    input_out: list[np.ndarray] = []
    cmos_out: list[np.ndarray] = []
    core_out: list[np.ndarray] = []
    maps: dict[str, np.ndarray] = {}
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows_cpu), batch_size):
            chunk = rows_cpu[start:start + batch_size]
            fields = input_fields(context, chunk)
            result = forward_batch(fields, phase, context, return_camera=True,
                                   aperture_mm=aperture_mm, lens_pitch_um=lens_pitch_um,
                                   screen_to_lens_um=screen_to_lens_um,
                                   lens_to_cmos_um=lens_to_cmos_um)
            intensity = result["camera_intensity"]
            scores = result["roi_scores"]
            detected = result["detected_energy"]
            core_energies = torch.stack(
                [(intensity * masks[index]).sum((1, 2)) * CMOS_PITCH_UM ** 2
                 for index in range(4)], dim=1)
            for offset in range(len(chunk)):
                label = int(labels_np[start + offset])
                statistic = row_statistics(intensity[offset], scores[offset], detected[offset],
                                           core_energies[offset], label)
                statistic["mnist_row"] = int(chunk[offset])
                records.append(statistic)
                if int(chunk[offset]) in save_set:
                    maps[str(int(chunk[offset]))] = intensity[offset].detach().cpu().numpy().astype(np.float32)
            scores_out.append(scores.detach().cpu().numpy().astype(np.float64))
            input_out.append(result["input_energy"].detach().cpu().double().numpy())
            cmos_out.append(detected.detach().cpu().double().numpy())
            core_out.append(core_energies.detach().cpu().double().numpy())
            if verbose and (start == 0 or start + batch_size >= len(rows_cpu)):
                print(f"eval[{tag}] {min(start + batch_size, len(rows_cpu))}/{len(rows_cpu)}",
                      flush=True)
    scores_np = np.concatenate(scores_out)
    input_np = np.concatenate(input_out)
    cmos_np = np.concatenate(cmos_out)
    core_np = np.concatenate(core_out)
    predictions = scores_np.argmax(1).astype(np.int64)
    row_ids = np.arange(len(labels_np))
    eta = cmos_np / np.maximum(input_np, EPS)
    core_fraction = core_np[row_ids, labels_np] / np.maximum(cmos_np, EPS)
    metrics = aggregate_statistics(records, labels_np, predictions, core_fraction, eta)
    metrics.update({
        "tag": tag,
        "identity": "C37_FIXED_ROI_OPTICAL_READOUT",
        "decision": "argmax of the four fixed ROI integrals; ties select the lowest ROI index",
        "core_fraction_of_roi_side": float(fraction),
        "aperture_mm": float(context["aperture_mm"] if aperture_mm is None else aperture_mm),
        "lens_plane_pitch_um": float(context["lens_pitch_um"] if lens_pitch_um is None
                                     else lens_pitch_um),
        "screen_to_lens_mm": float((context["screen_to_lens_um"] if screen_to_lens_um is None
                                    else screen_to_lens_um) / 1000.0),
        "lens_to_cmos_mm": float((context["lens_to_cmos_um"] if lens_to_cmos_um is None
                                  else lens_to_cmos_um) / 1000.0),
        "energy_accounting": "detected energy equals the four ROI integrals plus the "
                             "outside-ROI full-frame integral; no double normalisation",
        "zero_roi_rows": int(np.sum(scores_np.sum(1) <= 0)),
        "tied_maximum_rows": int(np.sum((scores_np == scores_np.max(1, keepdims=True)).sum(1) > 1)),
        "elapsed_seconds": float(time.perf_counter() - start_time),
    })
    return {
        "metrics": metrics,
        "records": records,
        "rows": rows_cpu.numpy().astype(np.int64),
        "labels": labels_np,
        "predictions": predictions,
        "roi_scores": scores_np,
        "core_energies": core_np,
        "input_energy": input_np,
        "cmos_energy": cmos_np,
        "eta": eta,
        "core_fraction": core_fraction,
        "maps": maps,
    }


def save_evaluation(path: Path, evaluation: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": evaluation["rows"],
        "labels": evaluation["labels"],
        "predictions": evaluation["predictions"],
        "roi_scores": evaluation["roi_scores"],
        "core_energies": evaluation["core_energies"],
        "input_energy_um2": evaluation["input_energy"],
        "cmos_energy_um2": evaluation["cmos_energy"],
        "eta_phase_to_cmos": evaluation["eta"],
        "core_fraction": evaluation["core_fraction"],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def balanced_rows(split_rows: torch.Tensor, labels: torch.Tensor, per_class: int,
                  *, offset: int = 0) -> torch.Tensor:
    selected = []
    for cls in range(4):
        part = split_rows[labels[split_rows] == cls]
        if len(part) < per_class + offset:
            raise ValueError(f"not enough rows for class {cls}")
        selected.append(part[offset:offset + per_class])
    return torch.cat(selected)


def derive_eta_target(config: dict[str, Any], p0_payload: dict[str, Any]) -> float:
    """Throughput target as a retention fraction of the frozen E0 mean efficiency."""
    energy = config["energy"]
    override = energy.get("eta_target_override")
    if override is not None:
        return float(override)
    retention = float(energy["eta_retention_fraction"])
    baseline = float(p0_payload["metrics"]["eta_phase_to_cmos"]["mean"])
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("P0 baseline efficiency is not usable for the throughput target")
    return retention * baseline
