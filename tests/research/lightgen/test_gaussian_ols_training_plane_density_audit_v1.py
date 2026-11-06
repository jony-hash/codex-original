from __future__ import annotations

import csv
import gzip
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from scripts.research.lightgen import execute_lightgen_matched_architecture_v1 as base


def _test_field(batch: int) -> ComplexField:
    plane = PlaneSpec("probe", (2, 2), (1.0e-6, 1.0e-6), 532.0e-9, 0.0, "test")
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.complex128).reshape(1, 1, 2, 2)
    return ComplexField(values.repeat(batch, 1, 1, 1), plane,
                        OperationProvenance("test", plane.plane_id, None))


class _ProbeRuntime:
    def __init__(self, batch: int) -> None:
        self.ols = SimpleNamespace(basis_id=base.GAUSSIAN_OLS_MODEL)
        self.config = SimpleNamespace(propagation_grid_size=2048)
        self.physical_optics = None
        self.slm_plane = _test_field(1).plane
        self.batch = batch

    def forward_from_image(self, _images: torch.Tensor) -> SimpleNamespace:
        field = _test_field(self.batch)
        return SimpleNamespace(
            image_input=SimpleNamespace(slm_field=field),
            ols_input_field=field,
            physical_power_trace={"P_input_W": torch.full((self.batch,), 2.0e-4)},
            encoder_records=(field, field),
            generator_records=(field, field),
            generator_input_field=field,
            generator_output_field=field,
        )

    @staticmethod
    def input_relay(field: ComplexField) -> ComplexField:
        return field

    @staticmethod
    def output_relay(field: ComplexField) -> ComplexField:
        return field


def _probe_model(batch: int = 20) -> base.SmallMatchedAdapter:
    model = object.__new__(base.SmallMatchedAdapter)
    torch.nn.Module.__init__(model)
    model.reference_parameter = torch.nn.Parameter(torch.ones(1))
    model.runtime = _ProbeRuntime(batch)
    model.reference_parameter.grad = torch.tensor([3.0])
    return model


def test_training_probe_is_fixed_idempotent_and_does_not_change_training_state(tmp_path: Path) -> None:
    model = _probe_model()
    model.train()
    pixels = torch.zeros((20, 28, 28), dtype=torch.uint8)
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)
    dev_rows = torch.arange(20, dtype=torch.long)
    random.seed(71)
    np.random.seed(72)
    torch.manual_seed(73)
    random_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state()
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    grad_before = model.reference_parameter.grad.detach().clone()
    probe_images = torch.zeros((20, 1, 28, 28), dtype=torch.float32)
    with torch.inference_mode():
        output_before = model.runtime.forward_from_image(probe_images).image_input.slm_field.tensor.clone()
        loss_before = (output_before.abs().square().mean()).item()

    result = base._record_gaussian_plane_density_probe(
        model, pixels, labels, dev_rows, tmp_path, branch="D0", global_step=45_000,
        grid_size=2048,
    )

    assert result["status"] == "RECORDED"
    shard = tmp_path / "plane_power_density" / "D0" / "step_045000.csv.gz"
    with gzip.open(shard, "rt", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 200
    assert {row["plane_key"] for row in rows} == {
        "source_before_slm", "slm_output", "encoder_entrance", "encoder_layer_1_aperture",
        "encoder_layer_2_aperture", "ols_incident", "fiber_output_and_decoder_entry",
        "decoder_layer_1_aperture", "decoder_layer_2_aperture_and_exit", "camera_physical_roi",
    }
    assert all(row["data_role"] == "fixed_DEV_probe" and row["grid_role"] == "training_production" for row in rows)
    assert all(float(row["mean_intensity_geometric_w_cm2"]) == pytest.approx(
        float(row["mean_intensity_geometric_w_m2"]) / 1.0e4
    ) for row in rows)
    assert [module.training for module in model.modules()] == [True]
    assert all(torch.equal(model.state_dict()[key], value) for key, value in state_before.items())
    assert torch.equal(model.reference_parameter.grad, grad_before)
    assert random.getstate() == random_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0] and np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    with torch.inference_mode():
        output_after = model.runtime.forward_from_image(probe_images).image_input.slm_field.tensor.clone()
        loss_after = (output_after.abs().square().mean()).item()
    assert torch.equal(output_before, output_after)
    assert loss_before == loss_after

    repeated = base._record_gaussian_plane_density_probe(
        model, pixels, labels, dev_rows, tmp_path, branch="D0", global_step=45_000,
        grid_size=2048,
    )
    assert repeated["status"] == "REUSED_VERIFIED"
    coverage = base._validate_gaussian_plane_density_coverage(tmp_path, "D0", [45_000], 2048)
    assert coverage["status"] == "VERIFIED" and coverage["record_count"] == 200


def test_training_probe_rejects_incomplete_existing_shard(tmp_path: Path) -> None:
    model = _probe_model()
    pixels = torch.zeros((20, 28, 28), dtype=torch.uint8)
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)
    dev_rows = torch.arange(20, dtype=torch.long)
    base._record_gaussian_plane_density_probe(
        model, pixels, labels, dev_rows, tmp_path, branch="D0", global_step=45_000,
        grid_size=2048,
    )
    shard = tmp_path / "plane_power_density" / "D0" / "step_045000.csv.gz"
    with gzip.open(shard, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_row", "label", "plane_index", "global_step", "branch", "grid_size"])
        writer.writeheader()
        writer.writerow({"sample_row": 0, "label": 0, "plane_index": 0, "global_step": 45_000,
                         "branch": "D0", "grid_size": 2048})
    with pytest.raises(RuntimeError, match="incomplete or belongs to another probe"):
        base._record_gaussian_plane_density_probe(
            model, pixels, labels, dev_rows, tmp_path, branch="D0", global_step=45_000,
            grid_size=2048,
        )
