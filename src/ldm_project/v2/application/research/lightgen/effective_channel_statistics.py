from __future__ import annotations

from typing import Any

import numpy as np


def stratified_bootstrap(values: np.ndarray, labels: np.ndarray, resamples: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    labels = np.asarray(labels, dtype=np.int64)
    if not (values.shape[0] == labels.shape[0] and values.ndim == 2):
        raise RuntimeError("bootstrap values/labels shape mismatch")
    classes = np.unique(labels)
    if not (len(classes) == 10 and np.array_equal(classes, np.arange(10))):
        raise RuntimeError("bootstrap must preserve all ten classes")
    rng = np.random.default_rng(seed)
    result = np.zeros((resamples, values.shape[1]), dtype=np.float64)
    for label in classes:
        population = values[labels == label]
        if not (len(population) > 1):
            raise RuntimeError(f"bootstrap class {label} is too small")
        class_sum = np.zeros_like(result)
        block = 100
        for start in range(0, resamples, block):
            stop = min(resamples, start + block)
            selected = rng.integers(0, len(population), size=(stop - start, len(population)))
            class_sum[start:stop] = population[selected].mean(axis=1)
        result += class_sum / len(classes)
    return result[:, 0] if squeeze else result


def fdr_bh(p_values: np.ndarray, q: float) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def maximum_port_gauge(coefficients: np.ndarray) -> np.ndarray:
    pivot_index = np.abs(coefficients).argmax(axis=1)
    pivot = coefficients[np.arange(len(coefficients)), pivot_index]
    phase = np.ones_like(pivot, dtype=np.complex128)
    nonzero = np.abs(pivot) > 0.0
    phase[nonzero] = pivot[nonzero] / np.abs(pivot[nonzero])
    return coefficients * phase.conj()[:, None]


def complex_covariances(coefficients: np.ndarray) -> dict[str, np.ndarray]:
    z = np.asarray(coefficients, dtype=np.complex128)
    mean = z.mean(axis=0)
    centered = z - mean
    covariance = np.einsum("ni,nj->ij", centered, centered.conj(), optimize=True) / len(z)
    second = np.einsum("ni,nj->ij", z, z.conj(), optimize=True) / len(z)
    return {"mean": mean, "covariance": covariance, "second_moment": second}


def complex_from_interleaved_real_covariance(covariance: np.ndarray) -> np.ndarray:
    real = covariance[0::2, 0::2]
    imag = covariance[1::2, 1::2]
    real_imag = covariance[0::2, 1::2]
    imag_real = covariance[1::2, 0::2]
    return real + imag + 1j * (imag_real - real_imag)


def noise_population_summary(endpoint_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    exposures = (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
    channels = ("raw_zero", "equal_ols_power")
    for endpoint, data in endpoint_data.items():
        result[endpoint] = {}
        for control_index, control in enumerate(channels):
            result[endpoint][control] = {}
            for exposure_index, exposure in enumerate(exposures):
                counts = data["observable_port_counts"][:, control_index, exposure_index, :]
                result[endpoint][control][str(exposure)] = {
                    "mean_observable_roots_snr_ge_1": float(counts[:, 0].mean()),
                    "mean_observable_roots_snr_ge_3": float(counts[:, 1].mean()),
                    "mean_observable_roots_snr_ge_5": float(counts[:, 2].mean()),
                    "p10_observable_roots_snr_ge_3": float(np.quantile(counts[:, 1], 0.1)),
                    "median_observable_roots_snr_ge_3": float(np.median(counts[:, 1])),
                    "p90_observable_roots_snr_ge_3": float(np.quantile(counts[:, 1], 0.9)),
                    "per_class_mean_observable_roots_snr_ge_3": {
                        str(label): float(counts[data["labels"] == label, 1].mean()) for label in range(10)
                    },
                }
    return result
