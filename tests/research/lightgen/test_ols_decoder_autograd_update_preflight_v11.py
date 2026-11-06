from __future__ import annotations

import torch

from scripts.research.lightgen import execute_ols_decoder_autograd_update_preflight_v11 as v11


def _config() -> dict:
    return v11._load_config(v11.DEFAULT_CONFIG)


def test_camera_task_mapping_value_gradient_and_saturation() -> None:
    config = _config()
    camera = config["camera"]
    photon_energy = v11.H_PLANCK * v11.C_LIGHT / config["optics"]["wavelength_m"]
    gain = camera["quantum_efficiency"] * camera["primary_exposure_s"] / photon_energy / camera["full_well_e"]
    saturation_power = 1.0 / gain
    power = torch.tensor([0.25 * saturation_power, 2.0 * saturation_power], dtype=torch.float64, requires_grad=True)

    electrons, task = v11._camera_task(power, config)
    gradient = torch.autograd.grad(task.sum(), power)[0]

    assert torch.allclose(electrons, power * camera["quantum_efficiency"] * camera["primary_exposure_s"] / photon_energy)
    assert torch.allclose(task, torch.tensor([0.25, 1.0], dtype=torch.float64))
    assert torch.allclose(gradient, torch.tensor([gain, 0.0], dtype=torch.float64), rtol=1e-12, atol=1e-18)


def test_camera_task_one_sided_full_well_boundary() -> None:
    config = _config()
    camera = config["camera"]
    gain = camera["quantum_efficiency"] * camera["primary_exposure_s"] / (v11.H_PLANCK * v11.C_LIGHT / config["optics"]["wavelength_m"]) / camera["full_well_e"]
    boundary = torch.tensor([1.0 - 1e-6, 1.0, 1.0 + 1e-6], dtype=torch.float64) / gain
    _, task = v11._camera_task(boundary, config)
    left = float((task[1] - task[0]) / (boundary[1] - boundary[0]))
    right = float((task[2] - task[1]) / (boundary[2] - boundary[1]))

    assert abs(left - gain) <= 1e-4 * gain
    assert abs(right) <= 1e-10


def test_state_comparison_rejects_changed_adam_tensor() -> None:
    original = {"exp_avg": torch.zeros(4), "step": torch.tensor(1.0)}
    recovered = {"exp_avg": torch.tensor([0.0, 0.0, 1e-3, 0.0]), "step": torch.tensor(1.0)}

    assert v11._tree_compare(original, recovered, atol=1e-7, rtol=1e-6)
    assert not v11._tree_compare(original, original, atol=1e-7, rtol=1e-6)
