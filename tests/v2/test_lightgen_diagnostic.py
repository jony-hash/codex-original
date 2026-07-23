import json

import pytest
import torch

from ldm_project.v2.config import LightGenSystemConfig, load_config
from ldm_project.v2.lightgen_contract import LightGenReferenceContractV2
from ldm_project.v2.optics.lightgen import LightGenAmplitudeFiberOLSSystemV2
from ldm_project.v2.optics.fiber_ols import reference_project_single_mode_fibers
from ldm_project.v2.application.cli import _lightgen_seeds


def _small_config() -> LightGenSystemConfig:
    # Test-only construction exercises the isolated operator without relaxing the public 400x400 YAML contract.
    return LightGenSystemConfig.model_construct(
        kind="diagnostic_lightgen_amplitude_fiber_ols_400", slm_shape=(32, 32), slm_pixel_pitch_m=3e-6,
        metasurface_shape=(32, 32), metasurface_pixel_pitch_m=3e-6, propagation_grid_size=32,
        encoder_layers=2, decoder_layers=4, fiber_grid_shape=(10, 10), fiber_mode_waist_m=8e-6,
        proxy_wavelength_m=532e-9, proxy_layer_spacing_m=1e-4, ols_kind="gaussian_single_mode_fiber",
        boundary="finite_crop", physical_parameter_status="unknown",
    )


def test_lightgen_contract_preserves_unknown_physics():
    contract = LightGenReferenceContractV2.from_dict(json.loads(open("docs/v2/lightgen_reference_contract_v2.json", encoding="utf-8").read()))
    assert contract.architecture_proxy_only
    with pytest.raises(ValueError, match="unknown"):
        LightGenReferenceContractV2.from_dict({**contract.to_dict(), "fields": {**contract.to_dict()["fields"], "wavelength_m": {"value": 532e-9, "status": "unknown", "source": None}}})


def test_lightgen_public_config_locks_reference_geometry():
    config = load_config("configs/v2/diagnostic_lightgen_amplitude_fiber_ols_400.yaml")
    assert config.system.slm_shape == (400, 400)
    with pytest.raises(ValueError, match="locks slm_shape"):
        LightGenSystemConfig(kind="diagnostic_lightgen_amplitude_fiber_ols_400", slm_shape=(200, 200))


def test_fiber_ols_keeps_complex_interface_and_matches_reference():
    system = LightGenAmplitudeFiberOLSSystemV2(_small_config())
    seed = torch.full((2, 1, 32, 32), 0.5)
    before = {name: value.detach().clone() for name, value in system.state_dict().items()}
    result = system(system.command(seed))
    reference = reference_project_single_mode_fibers(result.optical_trace[3], result.ols.modes)
    assert result.ols.coefficients.shape == (2, 100)
    assert torch.is_complex(result.ols.coefficients)
    assert torch.allclose(result.ols.coefficients, reference)
    assert result.raw_camera.tensor.shape == (2, 1, 32, 32)
    assert all(torch.equal(value, system.state_dict()[name]) for name, value in before.items())


def test_formal_config_cannot_be_used_as_lightgen_audit():
    config = load_config("configs/v2/formal_amplitude_finite_crop_identity.yaml")
    assert config.system.kind == "formal_amplitude_finite_crop_identity"


def test_global_nonlinear_seed_keeps_eight_global_directions():
    seeds = _lightgen_seeds(device=torch.device("cpu"), shape=(32, 32))
    assert seeds.shape == (8, 1, 32, 32)
    assert torch.isfinite(seeds).all()
    assert not torch.allclose(seeds[0], seeds[1])
