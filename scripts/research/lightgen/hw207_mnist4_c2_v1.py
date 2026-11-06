"""C2 diagnostic binary micromirror geometry in the target-plane coordinates."""
import math
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

DMD_PITCH_UM = 7.56
TARGET_WIDTH_UM = 200.0


def geometry(mirrors, magnification):
    if mirrors <= 0 or magnification == 0 or not math.isfinite(magnification):
        raise ValueError("Expected positive mirror count and finite nonzero magnification")
    pitch = abs(magnification) * DMD_PITCH_UM
    return {"mirrors": mirrors, "magnification": magnification, "pitch_um": pitch,
            "array_width_um": mirrors * pitch, "target_width_um": TARGET_WIDTH_UM,
            "identity": "ILLUSTRATIVE_ASSUMPTION_ONLY"}


def encode(binary_images, mirrors, method, threshold=0.5, magnification=None):
    if binary_images.ndim != 3 or binary_images.shape[-2:] not in ((28, 28), (100, 100)):
        raise ValueError("Expected batches of 28x28 or 100x100 binary images")
    if not ((binary_images == 0) | (binary_images == 1)).all():
        raise ValueError("Input must be binary")
    if mirrors <= 0 or not 0 <= threshold <= 1:
        raise ValueError("Invalid mirror count or threshold")
    image = binary_images.float()[:, None]
    if magnification is not None:
        pitch = geometry(mirrors, magnification)["pitch_um"]
        source_size = image.shape[-1]
        pixel_width = TARGET_WIDTH_UM / source_size
        centers = (torch.arange(mirrors, device=image.device, dtype=torch.float64) + 0.5 - mirrors / 2) * pitch
        if method == "nearest":
            indices = torch.floor((centers + TARGET_WIDTH_UM / 2) / pixel_width).long()
            valid = (indices >= 0) & (indices < source_size)
            safe = indices.clamp(0, source_size - 1)
            mapped = image[:, :, safe[:, None], safe[None, :]] * (valid[:, None] & valid[None, :])
        elif method == "area":
            source_start = torch.arange(source_size, device=image.device, dtype=torch.float64) * pixel_width - TARGET_WIDTH_UM / 2
            left = centers - pitch / 2
            right = centers + pitch / 2
            overlap = (torch.minimum(right[:, None], source_start[None, :] + pixel_width)
                       - torch.maximum(left[:, None], source_start[None, :])).clamp_min(0) / pitch
            mapped = (overlap @ image.double() @ overlap.T)
        else:
            raise ValueError("Unsupported mapping method")
    elif method == "nearest":
        mapped = F.interpolate(image, size=(mirrors, mirrors), mode="nearest-exact")
    elif method == "area":
        source_size = image.shape[-1]
        source_start = torch.arange(source_size, device=image.device, dtype=torch.float32)
        target_start = torch.arange(mirrors, device=image.device, dtype=torch.float32) * source_size / mirrors
        target_end = target_start + source_size / mirrors
        overlap = (torch.minimum(target_end[:, None], source_start[None, :] + 1)
                   - torch.maximum(target_start[:, None], source_start[None, :])).clamp_min(0)
        overlap = overlap / (source_size / mirrors)
        mapped = (overlap @ image @ overlap.T)
    else:
        raise ValueError("Unsupported mapping method")
    return (mapped[:, 0] >= threshold).to(torch.float32)


def target_plane_samples(pattern, magnification, samples_per_pixel=8, fill_factor=1.0,
                         offset_um=(0.0, 0.0), rotation_rad=0.0):
    """Return a sampled field with ideal lossless magnification normalization."""
    if pattern.ndim != 3 or pattern.shape[-2] != pattern.shape[-1]:
        raise ValueError("Expected a batch of square patterns")
    if not ((pattern == 0) | (pattern == 1)).all():
        raise ValueError("Pattern must be binary")
    if samples_per_pixel < 1 or not 0 < fill_factor <= 1:
        raise ValueError("Invalid subpixel sampling or fill factor")
    g = geometry(pattern.shape[-1], magnification)
    n = 100 * samples_per_pixel
    coords = (torch.arange(n, device=pattern.device, dtype=torch.float64) + 0.5) * (200.0 / n) - 100.0
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    x = x - offset_um[0]
    y = y - offset_um[1]
    c, s = math.cos(rotation_rad), math.sin(rotation_rad)
    source_x = (c * x + s * y) / magnification
    source_y = (-s * x + c * y) / magnification
    size = pattern.shape[-1]
    column = torch.floor(source_x / DMD_PITCH_UM + size / 2).long()
    row = torch.floor(source_y / DMD_PITCH_UM + size / 2).long()
    valid = (row >= 0) & (row < size) & (column >= 0) & (column < size)
    local_x = source_x - (column + 0.5 - size / 2) * DMD_PITCH_UM
    local_y = source_y - (row + 0.5 - size / 2) * DMD_PITCH_UM
    half = DMD_PITCH_UM * math.sqrt(fill_factor) / 2
    valid &= (local_x.abs() < half) & (local_y.abs() < half)
    field = pattern[:, row.clamp(0, size - 1), column.clamp(0, size - 1)] * valid
    return field.float() / abs(magnification), g


def target_field(pattern, magnification, samples_per_pixel=8, fill_factor=1.0,
                 offset_um=(0.0, 0.0), rotation_rad=0.0):
    """Return pixel-averaged field; resolve energy on the subpixel grid."""
    field, g = target_plane_samples(pattern, magnification, samples_per_pixel,
                                    fill_factor, offset_um, rotation_rad)
    return F.avg_pool2d(field[:, None], samples_per_pixel)[:, 0], g


def axis_aligned_pixel_integrals(pattern, magnification, fill_factor=1.0, offset_um=(0.0, 0.0)):
    """Exact pixel averages for square mirror regions without rotation.

    Return coherent field average and mean intensity separately. Squaring the
    average field discards unresolved subpixel energy and must not be used as
    the pixel-integrated intensity.
    """
    if pattern.ndim != 3 or pattern.shape[-2] != pattern.shape[-1]:
        raise ValueError("Expected a batch of square patterns")
    if not ((pattern == 0) | (pattern == 1)).all():
        raise ValueError("Pattern must be binary")
    if not 0 < fill_factor <= 1 or len(offset_um) != 2 or not all(math.isfinite(v) for v in offset_um):
        raise ValueError("Invalid fill factor or offset")
    count = pattern.shape[-1]
    g = geometry(count, magnification)
    pitch = g["pitch_um"]
    half_width = pitch * math.sqrt(fill_factor) / 2
    dtype = torch.float64
    pixel_start = torch.arange(100, dtype=dtype, device=pattern.device) * 2 - 100
    centers = (torch.arange(count, dtype=dtype, device=pattern.device) + 0.5 - count / 2) * pitch
    matrices = []
    for offset in offset_um:
        left = centers + offset - half_width
        right = centers + offset + half_width
        overlap = (torch.minimum(pixel_start[:, None] + 2, right[None, :])
                   - torch.maximum(pixel_start[:, None], left[None, :])).clamp_min(0)
        matrices.append(overlap / 2)
    image = pattern.double()
    if magnification < 0:
        image = image.flip((-2, -1))
    x_overlap, y_overlap = matrices
    coverage = y_overlap @ image @ x_overlap.T
    return {"mean_field": coverage / abs(magnification),
            "mean_intensity": coverage / magnification ** 2,
            "coordinate_identity": "ASSUMED_AXIS_ALIGNED", "geometry": g}


def energy_account(pattern, sampled_field, magnification, fill_factor):
    """Return ideal relative power on the DMD and sampled target planes in µm²."""
    if pattern.ndim != 3 or sampled_field.ndim != 3 or len(pattern) != len(sampled_field):
        raise ValueError("Expected matching pattern and target batches")
    g = geometry(pattern.shape[-1], magnification)
    if sampled_field.shape[-1] != sampled_field.shape[-2] or not 0 < fill_factor <= 1:
        raise ValueError("Invalid target grid or fill factor")
    pixel_area = (TARGET_WIDTH_UM / sampled_field.shape[-1]) ** 2
    incident = torch.full((len(pattern),), pattern.shape[-1] ** 2 * DMD_PITCH_UM ** 2,
                          dtype=torch.float64, device=pattern.device)
    on_area = pattern.double().sum((-2, -1)) * DMD_PITCH_UM ** 2
    active_area = on_area * fill_factor
    target = sampled_field.abs().double().square().sum((-2, -1)) * pixel_area
    return {"incident_um2": incident, "on_um2": on_area, "active_um2": active_area,
            "target_window_um2": target, "target_pixel_area_um2": pixel_area,
            "identity": g["identity"]}


def coherent_pupil(field, pitch_um, cutoff_cycles_um, defocus_um=0.0, wavelength_um=0.532):
    """Apply an assumed circular coherent pupil and angular-spectrum defocus phase."""
    if field.ndim != 3 or not field.is_complex() or pitch_um <= 0 or cutoff_cycles_um <= 0:
        raise ValueError("Expected complex field and positive pitch and cutoff")
    if not all(math.isfinite(x) for x in (pitch_um, cutoff_cycles_um, defocus_um, wavelength_um)) or wavelength_um <= 0:
        raise ValueError("Optical parameters must be finite")
    fy = torch.fft.fftfreq(field.shape[-2], d=pitch_um, device=field.device, dtype=field.real.dtype)
    fx = torch.fft.fftfreq(field.shape[-1], d=pitch_um, device=field.device, dtype=field.real.dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius2 = xx.square() + yy.square()
    propagating = 1 - wavelength_um ** 2 * radius2
    pupil = (radius2 <= cutoff_cycles_um ** 2) & (propagating >= 0)
    phase = (2 * math.pi * defocus_um / wavelength_um) * propagating.clamp_min(0).sqrt()
    transfer = torch.polar(pupil.to(field.real.dtype), phase)
    return torch.fft.ifft2(torch.fft.fft2(field, norm="ortho") * transfer, norm="ortho")


def calibration_patterns(mirrors):
    """Return distinct binary calibration commands on the micromirror lattice."""
    if mirrors < 8:
        raise ValueError("At least eight mirrors are required for calibration patterns")
    y, x = torch.meshgrid(torch.arange(mirrors), torch.arange(mirrors), indexing="ij")
    zero = torch.zeros((mirrors, mirrors), dtype=torch.float32)
    asymmetric = zero.clone()
    asymmetric[1:5, 1] = 1
    asymmetric[4, 1:4] = 1
    points = zero.clone()
    points[2, 2] = points[2, mirrors - 3] = points[mirrors - 3, 3] = 1
    isolated = zero.clone()
    isolated[mirrors // 2, mirrors // 2] = 1
    adjacent = isolated.clone()
    adjacent[mirrors // 2, mirrors // 2 + 1] = 1
    patterns = {"all_off": zero, "all_on": torch.ones_like(zero),
                "checkerboard": ((x + y) % 2).float(),
                "asymmetric": asymmetric, "fiducials": points,
                "isolated": isolated, "adjacent": adjacent}
    for period in (2, 4, 8):
        patterns[f"vertical_period_{period}"] = ((x % period) < period // 2).float()
        patterns[f"horizontal_period_{period}"] = ((y % period) < period // 2).float()
    return patterns


def intensity_metrics(reference, candidate):
    """Compare equal-coordinate intensity images without changing their energy."""
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError("Expected matching batches of 2-D images")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("Intensity must be finite")
    if (reference < 0).any() or (candidate < 0).any():
        raise ValueError("Intensity must be nonnegative")
    ref = reference.flatten(1).double()
    out = candidate.flatten(1).double()
    ref_norm = torch.linalg.vector_norm(ref, dim=1)
    out_norm = torch.linalg.vector_norm(out, dim=1)
    denominator = ref_norm * out_norm
    nrmse = torch.where(ref_norm > 0, torch.linalg.vector_norm(ref - out, dim=1) / ref_norm,
                        torch.full_like(ref_norm, float("nan")))
    correlation = torch.where(denominator > 0, (ref * out).sum(1) / denominator,
                              torch.full_like(ref_norm, float("nan")))
    return {"nrmse": nrmse, "normalized_correlation": correlation,
            "reference_sum": ref.sum(1), "candidate_sum": out.sum(1)}


def bind_identity_binary_heads(readout_root, device):
    """Bind C1 Identity Binary frozen heads and their train-only feature scales."""
    root = Path(readout_root)
    selection = json.loads((root / "readout_selection.json").read_text(encoding="utf-8"))
    heads = {}
    for mode in ("raw", "normalized"):
        key = f"binary/0/100/{mode}"
        choice = selection[key]
        scale = float(choice["scale"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid C1 scale: {key}")
        models = []
        for seed in (11, 23, 37):
            path = root / "readouts" / "binary" / "0" / "100" / mode / f"decay_{choice['decay']}_seed_{seed}.pt"
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if (saved["seed"] != seed or saved["decay"] != choice["decay"]
                    or saved["scale"] != scale or len(saved["history"]) != 30):
                raise ValueError(f"C1 model metadata mismatch: {path}")
            state = saved["state"]
            if state["weight"].shape != (4, 10000) or state["bias"].shape != (4,):
                raise ValueError(f"C1 model dimension mismatch: {path}")
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise ValueError(f"Nonfinite C1 model: {path}")
            model = torch.nn.Linear(10000, 4, device=device)
            model.load_state_dict(state)
            model.eval()
            model.requires_grad_(False)
            models.append({"seed": seed, "sha256": digest, "path": str(path.resolve()), "model": model})
        heads[mode] = {"scale": scale, "decay": choice["decay"], "models": models}
    return heads


def frozen_scores(intensity, head, normalize):
    """Evaluate C1-compatible raw or per-image-energy-normalized intensity."""
    if intensity.ndim != 3 or intensity.shape[-2:] != (100, 100):
        raise ValueError("Expected batch of 100x100 target intensities")
    if not torch.isfinite(intensity).all() or (intensity < 0).any():
        raise ValueError("Intensity must be finite and nonnegative")
    energy = intensity.sum((-2, -1), keepdim=True)
    if normalize:
        intensity = intensity / torch.where(energy > 0, energy, torch.ones_like(energy))
    features = intensity.flatten(1) / head["scale"]
    outputs = []
    with torch.no_grad():
        for item in head["models"]:
            logits = item["model"](features.to(next(item["model"].parameters()).device))
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite frozen readout logits")
            top = logits.topk(2, dim=1)
            outputs.append({"seed": item["seed"], "prediction": logits.argmax(1),
                            "margin": top.values[:, 0] - top.values[:, 1]})
    return outputs, (energy.flatten() == 0)
