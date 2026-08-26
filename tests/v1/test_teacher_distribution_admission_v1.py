from __future__ import annotations

import torch

from ldm_project.v1.application.teacher_distribution_admission import (
    _config,
    _lineage,
    _noise,
    _sample_seed,
)


CONFIG = "configs/research/teacher_distribution_admission_v1.yaml"


def test_package_b_config_is_locked_and_lineage_is_resolvable() -> None:
    config = _config(CONFIG)
    lineage = _lineage(config)
    assert config["sampling_steps"] == 200
    assert config["microbatch_size"] == 16
    assert lineage["teacher_model_version"] == "physical_first_linear_metasurface_vae_v5"
    assert lineage["diffusion_model_version"] == "conditioned_ldm_unet_v1"
    assert lineage["first_stage_state_hash"]
    assert lineage["diffusion_state_hash"]


def test_package_b_noise_seeds_are_deterministic_and_full_latent_gaussian() -> None:
    config = _config(CONFIG)
    first = _sample_seed(config, bank=0, label=3, index=7)
    second = _sample_seed(config, bank=0, label=3, index=7)
    other_bank = _sample_seed(config, bank=1, label=3, index=7)
    assert first == second
    assert first != other_bank
    noise = _noise([first, other_bank], (1, 25, 25))
    assert noise.shape == (2, 1, 25, 25)
    assert torch.isfinite(noise).all()

