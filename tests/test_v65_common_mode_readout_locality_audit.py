from __future__ import annotations

import importlib.util

import torch


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "diagnose_v65_common_mode_readout_locality_audit.py"


def _module():
    spec = importlib.util.spec_from_file_location("v65_locality", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _images(count: int) -> torch.Tensor:
    return (
        torch.arange(count * 200 * 200, dtype=torch.float32).reshape(count, 1, 200, 200)
        / 100000.0
    )


def test_common_mode_uses_train_only_mean_and_fixed_gain():
    module = _module()
    train = _images(4)
    first, params = module.fixed_common_mode_display(train, _images(2))
    second, params_again = module.fixed_common_mode_display(train, _images(2) + 10.0)
    assert params == params_again


def test_pointwise_affine_recovers_synthetic_native_mapping():
    module = _module()
    train = _images(4)
    val = _images(2) + 0.01
    target_train = 2.0 * train + 0.1
    target_val = 2.0 * val + 0.1
    report, model = module._pointwise_fit(train, target_train, val, target_val)
    prediction = module._pointwise_predict(model, val)
    assert report["probe"]["kind"] == "held_out_pointwise_affine"
    assert float((prediction - target_val).abs().mean()) < 1.0e-3


def test_local_convolution_recovers_center_pixel_without_spatial_resize():
    module = _module()
    train = _images(4)
    val = _images(2) + 0.01
    report, weight = module._conv_fit(
        train, train * 3.0, val, val * 3.0, 3, torch.device("cpu")
    )
    prediction = module._conv_predict(val, weight, 3, torch.device("cpu"))
    assert report["probe"]["kernel_size"] == 3
    assert float((prediction - val * 3.0).abs().mean()) < 1.0e-3


def test_fourier_observer_is_deterministic_on_native_images():
    module = _module()
    train = _images(4)
    val = _images(2) + 0.01
    report_a, transfer_a = module._fourier_fit(
        train, train * 0.5, val, val * 0.5, torch.device("cpu")
    )
    report_b, transfer_b = module._fourier_fit(
        train, train * 0.5, val, val * 0.5, torch.device("cpu")
    )
    assert (
        report_a["probe"]["selected_penalty"] == report_b["probe"]["selected_penalty"]
    )
    assert torch.equal(transfer_a, transfer_b)


def _result(corr: float, dice: float, ci: bool) -> dict:
    return {
        "target_corr_mean": corr,
        "thresholded_dice_mean": dice,
        "prototype_label_accuracy": 1.0,
        "paired_rotation": {"ci_low_positive": ci},
    }


def test_decision_covers_all_registered_gates():
    module = _module()
    baseline = {"target_corr_mean": 0.945, "thresholded_dice_mean": 0.883}
    common = {
        "target_corr_mean": 0.8,
        "thresholded_dice_mean": 0.75,
        "prototype_label_accuracy": 0.8,
        "paired_rotation": {"ci_low_positive": True},
    }
    results = {
        "common_mode_display": common,
        "pointwise_affine": _result(0.1, 0.1, False),
        "fourier_wiener": _result(0.1, 0.1, False),
    }
    for size in module.KERNEL_SIZES:
        results[f"local_conv_{size}"] = _result(0.1, 0.1, False)
    assert module.decide(results, baseline)["status"] == "common-mode-display-limited"
    results["common_mode_display"] = _result(0.1, 0.1, False)
    results["pointwise_affine"] = _result(0.86, 0.81, True)
    assert (
        module.decide(results, baseline)["status"]
        == "photometric-local-readout-candidate"
    )
    results["pointwise_affine"] = _result(0.1, 0.1, False)
    results["local_conv_7"] = _result(0.86, 0.81, True)
    assert module.decide(results, baseline)["smallest_kernel_size"] == 7
    results["local_conv_7"] = _result(0.1, 0.1, False)
    results["fourier_wiener"] = _result(0.86, 0.81, True)
    assert (
        module.decide(results, baseline)["status"]
        == "translation-equivariant-global-readout-candidate"
    )
    results["fourier_wiener"] = _result(0.1, 0.1, False)
    assert (
        module.decide(results, baseline)["status"]
        == "nonlocal-codebook-objective-representation-mismatch-confirmed"
    )


def test_source_forbids_resize_learning_and_target_bridge_paths():
    source = TOOL.read_text(encoding="utf-8").lower()
    assert "interpolate(" not in source
    assert "adaptive_avg_pool" not in source
    assert "grid_sample" not in source
    assert "pca" not in source
    assert "optimizer" not in source.replace("optimizer_created", "")
    assert "target_code_bridge" not in source
