"""C1 input and angular-spectrum validation for HW207 numerical diagnostics."""
import argparse
import json
import math
import os
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c0_v1 import features, load, map_image
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.propagation import angular_spectrum_propagate, reference_angular_spectrum_propagate


def population(indices_path, data_path, train_path, test_path):
    indices = json.loads(indices_path.read_text(encoding="utf-8"))
    data = json.loads(data_path.read_text(encoding="utf-8"))
    train_x, train_y = load(train_path, 60000)
    test_x, test_y = load(test_path, 10000)
    result = {}
    seen = set()
    for role, x, y, expected_source in (
        ("train", train_x, train_y, "training.pt"),
        ("validation", train_x, train_y, "training.pt"),
        ("test", test_x, test_y, "test.pt"),
    ):
        entry = indices[role]
        if Path(entry["source"]).name != expected_source:
            raise ValueError(f"Incorrect source role: {role}")
        rows = torch.tensor(entry["original_indices"], dtype=torch.int64)
        if rows.numel() != {"train": 22279, "validation": 2475, "test": 4157}[role]:
            raise ValueError(f"Incorrect sample count: {role}")
        if (rows < 0).any() or (rows >= len(y)).any() or rows.unique().numel() != rows.numel():
            raise ValueError(f"Invalid or repeated indices: {role}")
        keys = {(expected_source, int(i)) for i in rows}
        if seen.intersection(keys):
            raise ValueError(f"Overlapping source rows: {role}")
        seen.update(keys)
        labels = y[rows]
        counts = torch.bincount(labels, minlength=10).tolist()
        if counts[:4] != data["counts"][role] or any(counts[4:]):
            raise ValueError(f"Incorrect class counts: {role}")
        result[role] = (x[rows], labels, rows)
    return result


def propagate(amplitude, distance_m, window):
    if amplitude.shape[-2:] != (100, 100) or window < 100 or window % 2:
        raise ValueError("Expected 100x100 input and an even window of at least 100")
    if amplitude.dtype not in (torch.float32, torch.float64):
        raise ValueError("Amplitude must be float32 or float64")
    complex_dtype = torch.complex128 if amplitude.dtype == torch.float64 else torch.complex64
    if distance_m == 0:
        return amplitude.to(complex_dtype)
    pad = (window - 100) // 2
    padded = F.pad(amplitude, (pad, pad, pad, pad)).to(complex_dtype)
    plane = PlaneSpec("input", (window, window), (2e-6, 2e-6), 532e-9, 0, "numerical_yx")
    field = ComplexField(padded, plane, OperationProvenance("input", None, None))
    return angular_spectrum_propagate(field, distance_m, "output").tensor


def roi_scores(intensity, inset):
    if intensity.shape[-2:] != (100, 100) or inset not in (0, 5):
        raise ValueError("Invalid intensity shape or ROI inset")
    return torch.stack([intensity[..., y + inset:y + 50 - inset, x + inset:x + 50 - inset].sum((-2, -1))
                        for y, x in ((0, 0), (0, 50), (50, 0), (50, 50))], dim=-1)


def center_intensity(field):
    size = field.shape[-1]
    start = (size - 100) // 2
    return field[..., start:start + 100, start:start + 100].abs().square()


def window_comparison(amplitude, distance_m, small, large):
    """Return per-sample crop and ROI differences normalized to input energy."""
    if large <= small or amplitude.shape[-2:] != (100, 100):
        raise ValueError("Expected increasing windows and 100x100 amplitude")
    energy = amplitude.square().sum((-2, -1))
    if not torch.isfinite(amplitude).all() or (energy <= 0).any():
        raise ValueError("Comparison requires finite nonzero inputs")
    with torch.no_grad():
        first = center_intensity(propagate(amplitude, distance_m, small))
        second = center_intensity(propagate(amplitude, distance_m, large))
        image_error = (first - second).abs().sum((-2, -1)) / energy
        roi_error = (roi_scores(first, 0) - roi_scores(second, 0)).abs().sum(-1) / energy
    return image_error, roi_error


def precision_comparison(amplitude, distance_m, window):
    """Compare complex fields and full-window energy at matched sampling."""
    with torch.no_grad():
        single = propagate(amplitude.float(), distance_m, window)
        double = propagate(amplitude.double(), distance_m, window)
        numerator = torch.linalg.vector_norm((single.to(torch.complex128) - double).flatten(1), dim=1)
        denominator = torch.linalg.vector_norm(double.flatten(1), dim=1)
        if (denominator == 0).any():
            raise ValueError("Precision comparison requires nonzero fields")
        complex_error = numerator / denominator
        single_energy = single.abs().square().double().sum((-2, -1))
        double_energy = double.abs().square().sum((-2, -1))
        energy_error = (single_energy - double_energy).abs() / double_energy
    return complex_error, energy_error


def convergence_preflight(samples, device):
    """Evaluate the frozen validation subset for each input and distance."""
    images, labels, rows = samples["validation"]
    chosen = torch.cat([torch.where(labels == label)[0][:16] for label in range(4)])
    if chosen.numel() != 64:
        raise ValueError("Validation subset must contain 16 samples per class")
    results = {}
    for binary in (False, True):
        name = "binary" if binary else "gray"
        images_100 = map_image(features(images[chosen], binary), binary).to(device)
        results[name] = {}
        for distance in (0.0005, 0.001, 0.002):
            comparisons = []
            selected = None
            for small, large in ((512, 1024), (1024, 2048)):
                image_errors, roi_errors = [], []
                for image in images_100.split(1):
                    image_error, roi_error = window_comparison(image, distance, small, large)
                    image_errors.extend(image_error.cpu().tolist())
                    roi_errors.extend(roi_error.cpu().tolist())
                passed = max(image_errors + roi_errors) <= 1e-3
                comparisons.append({"small": small, "large": large, "image_errors": image_errors,
                                    "roi_errors": roi_errors, "passed": passed})
                if passed:
                    selected = large
                    break
            precision = None
            if selected is not None:
                complex_errors, energy_errors = [], []
                for image in images_100.split(1):
                    complex_error, energy_error = precision_comparison(image, distance, selected)
                    complex_errors.extend(complex_error.cpu().tolist())
                    energy_errors.extend(energy_error.cpu().tolist())
                precision = {"complex_errors": complex_errors, "energy_errors": energy_errors,
                             "passed": max(complex_errors + energy_errors) <= 1e-5}
            results[name][str(distance)] = {"comparisons": comparisons, "window": selected,
                                            "precision": precision}
            print(f"{name} {distance}m window={selected} precision={precision is not None and precision['passed']}",
                  flush=True)
    return {"validation_original_indices": rows[chosen].tolist(), "labels": labels[chosen].tolist(),
            "results": results}


def produce_chunk(images, labels, rows, binary, distance_m, window, device, microbatch, role="preflight"):
    """Propagate one ordered population chunk, reducing an oversized GPU microbatch."""
    if len(images) != len(labels) or len(rows) != len(labels) or not 0 < len(rows) <= 256:
        raise ValueError("Invalid chunk population")
    if microbatch < 1:
        raise ValueError("Microbatch must be positive")
    outputs = []
    offset = 0
    while offset < len(rows):
        end = min(offset + microbatch, len(rows))
        try:
            with torch.no_grad():
                amplitude = map_image(features(images[offset:end], binary), binary).to(device)
                field = propagate(amplitude, distance_m, window)
                intensity = center_intensity(field)
                outputs.append({
                    "intensity": intensity.cpu(),
                    "input_energy": amplitude.square().sum((-2, -1)).cpu(),
                    "full_energy": field.abs().square().sum((-2, -1)).cpu(),
                })
            offset = end
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or microbatch == 1:
                raise
            microbatch = max(1, microbatch // 2)
            torch.cuda.empty_cache()
    intensity = torch.cat([part["intensity"] for part in outputs])
    if not torch.isfinite(intensity).all() or (intensity < 0).any():
        raise ValueError("Nonfinite or negative intensity")
    return {
        "binary": binary, "distance_m": distance_m, "window": window, "role": role,
        "rows": rows.cpu(), "labels": labels.cpu(), "intensity": intensity,
        "input_energy": torch.cat([part["input_energy"] for part in outputs]),
        "full_energy": torch.cat([part["full_energy"] for part in outputs]),
        "crop_energy": intensity.sum((-2, -1)), "roi_a": roi_scores(intensity, 0),
        "roi_b": roi_scores(intensity, 5),
    }


def save_chunk(path, chunk):
    """Commit a chunk atomically without overwriting any existing evidence."""
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as output:
        torch.save(chunk, output)
        output.flush()
        os.fsync(output.fileno())
    os.link(temporary, path)
    temporary.unlink()


def checked_chunk(path, binary, distance_m, window, labels, rows, role="preflight"):
    """Load a committed chunk only when it matches the expected population."""
    chunk = torch.load(path, map_location="cpu", weights_only=True)
    expected = (binary, distance_m, window, role)
    actual = (chunk.get("binary"), chunk.get("distance_m"), chunk.get("window"), chunk.get("role"))
    if actual != expected or not torch.equal(chunk["labels"], labels.cpu()) or not torch.equal(chunk["rows"], rows.cpu()):
        raise ValueError(f"Chunk identity mismatch: {path}")
    n = len(rows)
    expected_shapes = {
        "intensity": (n, 100, 100), "input_energy": (n,), "full_energy": (n,),
        "crop_energy": (n,), "roi_a": (n, 4), "roi_b": (n, 4),
    }
    for name, shape in expected_shapes.items():
        value = chunk[name]
        if tuple(value.shape) != shape or not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"Invalid chunk {name}: {path}")
    if not torch.allclose(chunk["crop_energy"], chunk["intensity"].sum((-2, -1)), rtol=1e-6, atol=1e-5):
        raise ValueError(f"Chunk crop energy mismatch: {path}")
    for name, inset in (("roi_a", 0), ("roi_b", 5)):
        if not torch.allclose(chunk[name], roi_scores(chunk["intensity"], inset), rtol=1e-6, atol=1e-5):
            raise ValueError(f"Chunk ROI mismatch: {path}")
    return chunk


def numerical_probes(device):
    """Check the production ASM against independent complex-field references."""
    size, wavelength, distance = 64, 532e-9, 1e-3
    pitch_y, pitch_x = 3e-6, 2e-6
    plane = PlaneSpec("probe", (size, size), (pitch_y, pitch_x), wavelength, 0, "numerical_yx")
    provenance = OperationProvenance("input", None, None)
    generator = torch.Generator(device="cpu").manual_seed(207)
    real = torch.randn((size, size), generator=generator, dtype=torch.float64).to(device)
    imag = torch.randn((size, size), generator=generator, dtype=torch.float64).to(device)
    random_field = torch.complex(real, imag)
    field = ComplexField(random_field, plane, provenance)
    zero = angular_spectrum_propagate(field, 0, "zero").tensor
    zero_error = (torch.linalg.vector_norm(zero - random_field) / torch.linalg.vector_norm(random_field)).item()
    produced = angular_spectrum_propagate(field, distance, "output").tensor
    reference = reference_angular_spectrum_propagate(field, distance, "reference").tensor
    reference_error = (torch.linalg.vector_norm(produced - reference) / torch.linalg.vector_norm(reference)).item()
    single = angular_spectrum_propagate(ComplexField(random_field.to(torch.complex64), plane, provenance), distance, "fp32").tensor
    fp_error = (torch.linalg.vector_norm(single.to(torch.complex128) - produced) / torch.linalg.vector_norm(produced)).item()
    energy_error = ((single.abs().square().sum().double() - produced.abs().square().sum()).abs() / produced.abs().square().sum()).item()
    y = torch.arange(size, device=device, dtype=torch.float64)[:, None]
    x = torch.arange(size, device=device, dtype=torch.float64)[None, :]
    wave = torch.polar(torch.ones((size, size), device=device, dtype=torch.float64), 2 * math.pi * (3 * y / size + 7 * x / size))
    actual = angular_spectrum_propagate(ComplexField(wave, plane, provenance), distance, "wave").tensor
    phase = 2 * math.pi * distance / wavelength * math.sqrt(1 - (wavelength * 3 / (size * pitch_y)) ** 2 - (wavelength * 7 / (size * pitch_x)) ** 2)
    expected = wave * complex(math.cos(phase), math.sin(phase))
    opposite = wave * complex(math.cos(phase), -math.sin(phase))
    swapped_phase = 2 * math.pi * distance / wavelength * math.sqrt(1 - (wavelength * 3 / (size * pitch_x)) ** 2 - (wavelength * 7 / (size * pitch_y)) ** 2)
    swapped = wave * complex(math.cos(swapped_phase), math.sin(swapped_phase))
    norm = torch.linalg.vector_norm(wave)
    # Apply faults to the production output and feed them to the same checker.
    def wave_error(output):
        return (torch.linalg.vector_norm(output - expected) / norm).item()

    reverse_fault = actual * complex(math.cos(2 * phase), -math.sin(2 * phase))
    axis_delta = swapped_phase - phase
    axis_fault = actual * complex(math.cos(axis_delta), math.sin(axis_delta))
    low_wave = torch.polar(torch.ones_like(x).expand(size, size), 2 * math.pi * x.expand(size, size) / size)
    low_actual = angular_spectrum_propagate(
        ComplexField(low_wave, plane, provenance), distance, "fresnel_probe"
    ).tensor
    fresnel_phase = (2 * math.pi * distance / wavelength
                     - math.pi * wavelength * distance * (1 / (size * pitch_x)) ** 2)
    fresnel = low_wave * complex(math.cos(fresnel_phase), math.sin(fresnel_phase))
    errors = {"zero_complex_l2": zero_error, "reference_l2": reference_error,
              "fp32_complex_l2": fp_error, "fp32_energy": energy_error,
              "plane_wave_l2": (torch.linalg.vector_norm(actual - expected) / norm).item(),
              "opposite_phase_l2": (torch.linalg.vector_norm(actual - opposite) / norm).item(),
              "swapped_axes_l2": (torch.linalg.vector_norm(actual - swapped) / norm).item(),
              "production_reverse_fault_l2": wave_error(reverse_fault),
              "production_axis_fault_l2": wave_error(axis_fault),
              "fresnel_l2": (torch.linalg.vector_norm(low_actual - fresnel) / norm).item()}
    errors["passed"] = all(errors[k] <= 1e-5 for k in ("zero_complex_l2", "reference_l2", "fp32_complex_l2", "fp32_energy", "plane_wave_l2", "fresnel_l2")) and all(errors[k] > 1e-5 for k in ("production_reverse_fault_l2", "production_axis_fault_l2"))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--convergence-output", type=Path)
    for name in ("indices", "data", "train", "test"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    samples = population(args.indices, args.data, args.train, args.test)
    sample = samples["validation"][0][:1]
    mapped = map_image(features(sample, True), True)
    intensity = propagate(mapped, 0, 100).abs().square()
    probes = numerical_probes(torch.device(args.device))
    print(json.dumps({"population_counts": {k: len(v[1]) for k, v in samples.items()},
                      "identity_roi_a": roi_scores(intensity, 0).tolist(),
                      "numerical_probes": probes,
                      "status": "INPUT_AND_NUMERICAL_PROBES_ONLY"}))
    if not probes["passed"]:
        raise SystemExit("Numerical probes failed")
    if args.convergence_output is not None:
        if args.convergence_output.exists():
            raise FileExistsError(args.convergence_output)
        result = convergence_preflight(samples, torch.device(args.device))
        with args.convergence_output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
