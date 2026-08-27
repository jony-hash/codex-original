from __future__ import annotations

import torch
import pytest

from ldm_project.v1.application import package_b3_vae as producer
from ldm_project.v1.teacher.software_vae import (
    LATENT_SHAPE,
    MODEL_VERSION,
    SoftwareMultiscaleTeacherVAE,
    reconstruction_loss,
    state_content_hash,
)


CONFIG = "configs/research/teacher_first_stage_recovery_v1.yaml"


def test_b3_config_and_model_contract_are_locked() -> None:
    config = producer._config(CONFIG)
    assert config["model_version"] == MODEL_VERSION
    assert tuple(config["latent_shape"]) == LATENT_SHAPE
    assert config["formal_bank_partition_seed"] == 32452843
    assert config["formal_evaluator_in_producer"] is False
    model = SoftwareMultiscaleTeacherVAE()
    images = torch.rand(2, 1, 200, 200)
    output = model(images, deterministic=True)
    assert tuple(output["mu"].shape[1:]) == LATENT_SHAPE
    assert tuple(output["reconstruction"].shape[1:]) == (1, 200, 200)


def test_b3_decoder_has_no_label_shortcut_and_reconstruction_loss_updates() -> None:
    model = SoftwareMultiscaleTeacherVAE()
    images = torch.rand(4, 1, 200, 200)
    labels = torch.tensor([0, 1, 2, 3])
    output = model(images, deterministic=False)
    loss, metrics = reconstruction_loss(
        output["reconstruction"], images, labels, output["mu"], output["logvar"]
    )
    before = state_content_hash(model.state_dict())
    loss.backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    optimizer.step()
    assert torch.isfinite(loss)
    assert metrics["total"] == pytest.approx(float(loss.detach()))
    assert state_content_hash(model.state_dict()) != before
    assert not any(name.startswith("label") for name, _ in model.named_parameters())


def test_b3_formal_bank_partition_is_disjoint_and_reproducible() -> None:
    labels = torch.arange(10).repeat_interleave(500)
    images = torch.zeros(5000, 1, 28, 28)
    images[:, 0, 0, 0] = torch.arange(5000).float()
    data = {
        "images": images,
        "labels": labels,
        "formal_indices": torch.arange(5000),
    }
    config = producer._config(CONFIG)
    first = producer._formal_banks(data, config)
    second = producer._formal_banks(data, config)
    for (_, first_images, first_labels, first_sources), (_, second_images, second_labels, second_sources) in zip(first, second, strict=True):
        assert torch.equal(first_images, second_images)
        assert torch.equal(first_labels, second_labels)
        assert torch.equal(first_sources, second_sources)
    first_ids = set(first[0][3].tolist())
    second_ids = set(first[1][3].tolist())
    assert first_ids.isdisjoint(second_ids)


def test_b3_checkpoint_loader_rejects_wrong_model_version(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"model_version": "old_linear_metasurface"}, path)
    with pytest.raises(ValueError, match="checkpoint payload"):
        producer._load_checkpoint(path, torch.device("cpu"))


def test_b3_camera_plane_transform_is_deterministic_and_bounded() -> None:
    images = torch.rand(2, 1, 28, 28)
    first = producer._to_camera(images)
    second = producer._to_camera(images.clone())
    assert tuple(first.shape) == (2, 1, 200, 200)
    assert torch.equal(first, second)
    assert first.dtype == torch.float32
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0
