"""HW207 C3.3 single-layer trainable phase screen diagnostic; not a hardware model.

The only optical difference from C3.1/C3.2 is a trainable phase on the existing
single-layer screen plus an explicit camera-plane defocus.  Training and
preflight both call ``forward_field``; nothing else is optimised.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.research.lightgen.hw207_mnist4_c2_v1 import coherent_pupil
from scripts.research.lightgen.hw207_mnist4_c31_c32_cmos_v1 import prepare_configuration
from scripts.research.lightgen.hw207_mnist4_c31_c32_run_v1 import metrics
from ldm_project.v2.application.research.lightgen import hw207_spatial_geometry as geo
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from ldm_project.v2.optics.propagation import angular_spectrum_propagate

WAVELENGTH_M = 532e-9
PITCH_M = 2e-6
GRID = 100
CANDIDATE_INDEX = 2  # C3-C, the candidate used by the C3.1 phase preflight
ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/research/lightgen/hw207_mnist4_c21_c3_spatial_v1.json"


# --------------------------------------------------------------------------- #
# configuration and forward
# --------------------------------------------------------------------------- #
@dataclass
class Configuration:
    base: object
    defocus_um: float
    device: torch.device


def build_configuration(defocus_um: float, device: str = "cuda") -> Configuration:
    candidates = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["relay_candidates"]
    base = prepare_configuration(candidates[CANDIDATE_INDEX], torch.device(device), float(defocus_um))
    base.screen.requires_grad_(True)
    with torch.no_grad():
        base.screen.phase.zero_()
    return Configuration(base=base, defocus_um=float(defocus_um), device=torch.device(device))


def phase_parameter(configuration: Configuration) -> torch.nn.Parameter:
    return configuration.base.screen.phase


def forward_field(image: torch.Tensor, configuration: Configuration, phase: torch.Tensor | None = None):
    """Single forward path used by both preflight and training.

    ``image`` is uint8 [B,28,28].  ``phase`` overrides the module parameter.
    Returns the same dictionary layout as the C3.1 forward.
    """
    c = configuration.base
    if image.ndim != 3 or image.shape[-2:] != (28, 28) or image.dtype != torch.uint8:
        raise ValueError("image must be uint8 [batch,28,28]")
    binary = (image.to(c.device) >= 128).float()
    mapped = geo.map_binary_images(binary, c.source_matrix, c.target_matrix, c.affine)
    amplitude = mapped["irradiance"].clamp_min(0).sqrt()
    field = torch.complex(amplitude, torch.zeros_like(amplitude))
    padded = F.pad(field, (206, 206, 206, 206))
    filtered = coherent_pupil(padded, 2.0, 0.1, 0.0, 0.532)
    chip_input = filtered[:, 206:306, 206:306]
    source = ComplexField(chip_input[:, None], c.input_plane,
                          OperationProvenance("ideal_input_relay_pupil_crop", None, None))

    if phase is None:
        chip_output, _, _ = c.screen(source, "c31_screen")
    else:
        transfer = torch.polar(torch.ones_like(phase), phase).to(chip_input.dtype)
        modulated = ComplexField(chip_input[:, None] * transfer,
                                 source.plane, OperationProvenance("phase_only_modulation", None, None))
        chip_output = ComplexField(modulated.tensor, c.relay.source,
                                   OperationProvenance("phase_only_identity_spacing", None, 0.0))

    image_field = c.relay(chip_output)
    padded_camera = F.pad(image_field.tensor, (206, 206, 206, 206))
    camera_plane = image_field.plane.with_plane("c31_camera_pad", shape=(512, 512))
    camera_field = angular_spectrum_propagate(
        ComplexField(padded_camera, camera_plane,
                     OperationProvenance("camera_zero_padding", image_field.plane.plane_id, None)),
        c.defocus_um * 1e-6, "c31_camera_defocused")
    crop = camera_field.tensor[:, 0, 128:384, 128:384]
    intensity = crop.abs().square()
    pixels = torch.sparse.mm(c.camera_matrix, intensity.flatten(1).double().T).T
    pixels = pixels.reshape(-1, 768, 768)
    scores = torch.stack([pixels[:, r0:r1, c0:c1].sum((1, 2)) for r0, r1, c0, c1 in c.rois], dim=1)
    total = pixels.sum((1, 2))
    pitch_um = abs(c.magnification) * 2
    ledger = {
        "mapped_input_energy_um2": mapped["irradiance"].double().sum((1, 2)) * 4,
        "lightgen_input_energy_um2": chip_input.abs().square().double().sum((1, 2)) * 4,
        "lightgen_output_energy_um2": chip_output.tensor.abs().square().double().sum((1, 2, 3)) * 4,
        "relay_output_energy_um2": image_field.tensor.abs().square().double().sum((1, 2, 3)) * pitch_um ** 2,
        "defocus_window_energy_um2": camera_field.tensor.abs().square().double().sum((1, 2, 3)) * pitch_um ** 2,
        "camera_crop_energy_um2": intensity.double().sum((1, 2)) * pitch_um ** 2,
        "detected_energy_um2": total,
    }
    return {
        "dmd_binary_pattern": mapped["commands"],
        "lightgen_input_field": chip_input,
        "lightgen_output_field": chip_output.tensor[:, 0],
        "camera_ideal_intensity": intensity,
        "cmos_pixel_integral": pixels,
        "roi_scores": scores,
        "outside_roi_energy": (total - scores.sum(1)).clamp_min(0),
        "detected_energy": total,
        "energy": ledger,
    }


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS = {
    "temperature": 20.0,
    "ce": 1.0,
    "margin": 0.5,
    "leakage": 0.05,
    "throughput": 0.1,
    "tv": 1e-3,
    "throughput_floor": 0.8,
}


def phase_tv(phase: torch.Tensor) -> torch.Tensor:
    return (phase[..., 1:, :] - phase[..., :-1, :]).abs().mean() + \
           (phase[..., :, 1:] - phase[..., :, :-1]).abs().mean()


def loss_from_result(result: dict, labels: torch.Tensor, weights: dict | None = None) -> torch.Tensor:
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    scores = result["roi_scores"]
    total = result["detected_energy"].clamp_min(1e-20)
    logits = w["temperature"] * scores / total[:, None]
    ce = F.cross_entropy(logits.float(), labels)
    rows = torch.arange(len(labels), device=scores.device)
    correct = scores[rows, labels]
    wrong = scores.clone()
    wrong[rows, labels] = -float("inf")
    margin = F.relu(1.0 - (correct - wrong.max(1).values) / total).mean()
    outside = result["outside_roi_energy"] / total
    input_energy = result["energy"]["lightgen_input_energy_um2"].clamp_min(1e-20)
    throughput = total / input_energy
    loss = (w["ce"] * ce + w["margin"] * margin + w["leakage"] * outside.mean()
            + w["throughput"] * F.relu(w["throughput_floor"] - throughput).square().mean())
    phase = result.get("phase")
    if phase is not None and w["tv"] > 0:
        loss = loss + w["tv"] * phase_tv(phase)
    return loss


def predictions_from_scores(scores: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    maximum = scores.max(1, keepdim=True).values
    tied = torch.isclose(scores, maximum, rtol=1e-12, atol=0).sum(1) != 1
    return torch.where((total > 0) & ~tied, scores.argmax(1), -1)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_mnist(root: Path = ROOT):
    images = (root / "data/MNIST/raw/train-images-idx3-ubyte").read_bytes()
    labels = (root / "data/MNIST/raw/train-labels-idx1-ubyte").read_bytes()
    if struct.unpack(">IIII", images[:16]) != (2051, 60000, 28, 28):
        raise ValueError("unexpected MNIST image header")
    if struct.unpack(">II", labels[:8]) != (2049, 60000):
        raise ValueError("unexpected MNIST label header")
    x = torch.from_numpy(np.frombuffer(images, dtype=np.uint8, offset=16).reshape(60000, 28, 28).copy())
    y = torch.from_numpy(np.frombuffer(labels, dtype=np.uint8, offset=8).copy()).long()
    return x, y


def c0_train_indices(root: Path = ROOT) -> torch.Tensor:
    path = root / "outputs/20261103_hw207_mnist4_c2_v1/runs/autodl_backup_c0_inputs_01/indices.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    train = torch.tensor(data["train"]["original_indices"])
    if len(train) != 22279:
        raise ValueError("unexpected C0 training index count")
    return train


def stratified_split(train_indices: torch.Tensor, labels: torch.Tensor, *, seed: int = 20261104,
                     n_holdout: int = 2000, n_validation: int = 2000):
    """Disjoint phase_train / phase_validation / phase_holdout, stratified by class."""
    generator = torch.Generator().manual_seed(seed)
    by_class = []
    for k in range(4):
        rows = train_indices[labels[train_indices] == k]
        permutation = torch.randperm(len(rows), generator=generator)
        by_class.append(rows[permutation])
    holdout = torch.cat([rows[: n_holdout // 4] for rows in by_class])
    validation = torch.cat([rows[n_holdout // 4: n_holdout // 4 + n_validation // 4] for rows in by_class])
    train = torch.cat([rows[n_holdout // 4 + n_validation // 4:] for rows in by_class])
    for part in (train, validation, holdout):
        part, _ = torch.sort(part)
    return train, validation, holdout


# --------------------------------------------------------------------------- #
# phase -> height -> mesh
# --------------------------------------------------------------------------- #
def wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    return torch.remainder(phase, 2 * math.pi)


def phase_to_height_um(phase_rad: torch.Tensor, delta_n: float) -> torch.Tensor:
    if delta_n <= 0:
        raise ValueError("delta_n must be positive")
    return wrap_phase(phase_rad) * WAVELENGTH_M / (2 * math.pi * delta_n) * 1e6


def height_two_pi_um(delta_n: float) -> float:
    if delta_n <= 0:
        raise ValueError("delta_n must be positive")
    return WAVELENGTH_M / delta_n * 1e6


def quantize_height(height: torch.Tensor, levels: int, max_height_um: float) -> torch.Tensor:
    if levels < 2:
        raise ValueError("levels must be >= 2")
    step = max_height_um / (levels - 1)
    return torch.clamp(torch.round(height / step) * step, 0.0, max_height_um)


def min_feature_open(height: torch.Tensor, radius: int) -> torch.Tensor:
    """Grey-scale morphological opening with a (2r+1) square; removes fine spikes."""
    if radius <= 0:
        return height
    shape = height.shape
    x = height.reshape(-1, 1, shape[-2], shape[-1])
    eroded = -F.max_pool2d(-x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    opened = F.max_pool2d(eroded, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return opened.reshape(shape)


def smooth_height(height: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return height
    radius = max(1, int(math.ceil(3 * sigma)))
    grid = torch.arange(-radius, radius + 1, dtype=height.dtype, device=height.device)
    kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    kernel = kernel / kernel.sum()
    shape = height.shape
    x = height.reshape(-1, 1, shape[-2], shape[-1])
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), kernel[None, None, :, None])
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), kernel[None, None, None, :])
    return x.reshape(shape)


def perturb_height(height: torch.Tensor, sigma_um: float, generator: torch.Generator) -> torch.Tensor:
    if sigma_um <= 0:
        return height
    noise = torch.randn(height.shape, generator=generator, dtype=height.dtype) * sigma_um
    return torch.clamp(height + noise, 0.0, None)


def height_to_phase(height_um: torch.Tensor, delta_n: float) -> torch.Tensor:
    return 2 * math.pi * delta_n * height_um * 1e-6 / WAVELENGTH_M


def heightmap_triangles(height_um: np.ndarray, *, substrate_um: float) -> tuple[np.ndarray, dict]:
    """Closed manifold triangulation of a relief layer on a flat substrate slab.

    Coordinates are micrometres.  ``+z`` points out of the substrate toward the
    incident light; the solid occupies ``-substrate_um <= z <= h(x, y)`` with a
    single closed boundary (top surface, bottom surface, four side walls).
    """
    ny, nx = height_um.shape
    xs = np.arange(nx, dtype=np.float64) * PITCH_M * 1e6
    ys = np.arange(ny, dtype=np.float64) * PITCH_M * 1e6
    gx, gy = np.meshgrid(xs, ys)
    top = np.stack([gx, gy, height_um], axis=-1)
    bottom = np.stack([gx, gy, np.full_like(gx, -substrate_um)], axis=-1)
    vertices = np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3)], axis=0)
    n = nx * ny
    top_idx = np.arange(n).reshape(ny, nx)
    bottom_idx = n + top_idx
    triangles: list[np.ndarray] = []

    def quad(a: int, b: int, c: int, d: int, outward: tuple[float, float, float]) -> None:
        points = vertices[[a, b, c, d]]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        if float(np.dot(normal, np.asarray(outward, dtype=np.float64))) < 0:
            points = points[::-1]
        triangles.append(points[[0, 1, 2]])
        triangles.append(points[[0, 2, 3]])

    for j in range(ny - 1):
        for i in range(nx - 1):
            quad(top_idx[j, i], top_idx[j, i + 1], top_idx[j + 1, i + 1], top_idx[j + 1, i], (0, 0, 1))
            quad(bottom_idx[j, i], bottom_idx[j + 1, i], bottom_idx[j + 1, i + 1], bottom_idx[j, i + 1], (0, 0, -1))
    for i in range(nx - 1):
        quad(top_idx[0, i], top_idx[0, i + 1], bottom_idx[0, i + 1], bottom_idx[0, i], (0, -1, 0))
        quad(top_idx[ny - 1, i + 1], top_idx[ny - 1, i], bottom_idx[ny - 1, i], bottom_idx[ny - 1, i + 1], (0, 1, 0))
        quad(top_idx[i + 1, 0], top_idx[i, 0], bottom_idx[i, 0], bottom_idx[i + 1, 0], (-1, 0, 0))
        quad(top_idx[i, nx - 1], top_idx[i + 1, nx - 1], bottom_idx[i + 1, nx - 1], bottom_idx[i, nx - 1], (1, 0, 0))
    stack = np.stack(triangles).reshape(-1, 3, 3)
    return stack, {"vertices": int(len(vertices)), "triangles": int(len(triangles)),
                   "extent_um": [float(xs.max()), float(ys.max()), float(height_um.max())],
                   "substrate_um": float(substrate_um)}


def check_mesh(triangles: np.ndarray, *, tol: float = 1e-9) -> dict:
    """Closed-manifold and size checks for a triangle soup in micrometres."""
    flat = triangles.reshape(-1, 3)
    keys = np.round(flat / tol).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    order = np.sort(directed, axis=1)
    _, counts = np.unique(order, axis=0, return_counts=True)
    degenerate = int(np.sum(np.all(faces[:, [0]] == faces[:, [1]], axis=1) |
                            np.all(faces[:, [1]] == faces[:, [2]], axis=1) |
                            np.all(faces[:, [0]] == faces[:, [2]], axis=1)))
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    volume = float(np.sum(np.sum(triangles[:, 0] * np.cross(triangles[:, 1], triangles[:, 2]), axis=1)) / 6.0)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, directed_counts = np.unique(directed, axis=0, return_counts=True)
    return {"closed": bool(np.all(counts == 2)), "nonmanifold_edges": int(np.sum(counts != 2)),
            "orientation_consistent": bool(np.all(directed_counts == 1)),
            "degenerate_faces": degenerate, "unique_vertices": int(len(unique)),
            "triangles": int(len(faces)), "area_um2": float(area.sum()), "volume_um3": abs(volume),
            "bbox_um": [[float(flat[:, k].min()), float(flat[:, k].max())] for k in range(3)]}


def write_binary_stl(path: Path, triangles: np.ndarray, *, header: str = "hw207-c33") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii")[:80].ljust(80, b" "))
        handle.write(struct.pack("<I", len(triangles)))
        for normal, triangle in zip(normals, triangles, strict=True):
            handle.write(struct.pack("<12fH", *normal, *triangle.reshape(-1), 0))


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    count = struct.unpack("<I", raw[80:84])[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError("STL length does not match its triangle count")
    data = np.frombuffer(raw[84:], dtype=np.uint8).reshape(count, 50)
    floats = data[:, :48].copy().view("<f4").reshape(count, 12)
    return floats[:, 3:].reshape(count, 3, 3).astype(np.float64), floats[:, :3].astype(np.float64)


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate(images: torch.Tensor, labels: torch.Tensor, configuration: Configuration,
             phase: torch.Tensor, batch_size: int = 128, weights: dict | None = None) -> dict:
    """Frozen-phase evaluation on the same forward path used for training."""
    was_training = configuration.base.screen.training
    configuration.base.screen.eval()
    scores_all, total_all, maps, losses = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            chunk = images[start: start + batch_size]
            result = forward_field(chunk, configuration, phase)
            if weights is not None:
                losses.append(float(loss_from_result({**result, "phase": None},
                                                     labels[start: start + batch_size], weights).detach()))
            scores_all.append(result["roi_scores"].double().cpu())
            total_all.append(result["detected_energy"].double().cpu())
            maps.append(result["camera_ideal_intensity"].double().cpu())
    if was_training:
        configuration.base.screen.train()
    scores = torch.cat(scores_all)
    total = torch.cat(total_all)
    predictions = predictions_from_scores(scores, total)
    summary = metrics(labels.cpu(), predictions, scores, total)
    summary["mean_detected_energy"] = float(total.mean())
    summary["roi_energy_matrix"] = scores.mean(0).tolist()
    return {"metrics": summary, "scores": scores, "total": total,
            "predictions": predictions, "intensity": torch.cat(maps),
            "loss": (sum(losses) / len(losses)) if losses else None}