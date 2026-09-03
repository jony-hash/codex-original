from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/visualization/render_scaled_chen_fiber_horizontal.py"
VISUALIZATION_SCRIPTS = ROOT / "scripts/visualization"
NATURE_SCRIPTS = Path(r"C:\Users\24567\.codex\skills\nature-figure\scripts")


@pytest.fixture(scope="module")
def renderer():
    sys.path.insert(0, str(NATURE_SCRIPTS))
    sys.path.insert(0, str(VISUALIZATION_SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "scaled_chen_fiber_visualization_test_module", SCRIPT
        )
        if spec is None or spec.loader is None:
            raise AssertionError("failed to load fiber visualization renderer")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(VISUALIZATION_SCRIPTS))
        sys.path.remove(str(NATURE_SCRIPTS))


def test_vis1_contract_locks_branch_panel_and_arm_order(renderer) -> None:
    assert renderer.TASK_ID == "H-LOMA-CBOGT-FIBER-VIS1"
    assert renderer.CLASSIFICATION == "descriptive_fiber_visualization_completed"
    assert renderer.UNIQUE_NEXT_ACTION == "IMPLEMENT_H_LOMA_CBOGT_FIBER_VIS1"
    assert renderer.EXPECTED_BRANCH == "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
    assert renderer.EXPECTED_START_HEAD == "4d2933cb5b49302dd2114d49b56c905ef9d10ca7"
    assert renderer.ARM_ORDER == ("SMF", "FMF", "MMF-M6")
    assert renderer.MMF_FAMILY_ORDER == (
        "LP01",
        "LP11a",
        "ETA1",
        "XI2",
        "XIETA",
        "ETA2",
    )
    assert renderer.MMF_SYNTHETIC_EXTRA_FAMILIES == ("ETA1", "XI2", "XIETA", "ETA2")


def test_registered_panel_is_reused_without_expansion(renderer) -> None:
    contract = {
        "sample_panel": {
            "source": "confirmation_panel_train_indices_55000_59999",
            "count": 10,
            "order": list(range(10)),
            "indices_by_label": {
                "0": 55006,
                "1": 55000,
                "2": 55001,
                "3": 55008,
                "4": 55004,
                "5": 55010,
                "6": 55012,
                "7": 55003,
                "8": 55002,
                "9": 55007,
            },
        }
    }
    panel = renderer._fixed_panel(contract)
    assert panel["count"] == 10
    assert panel["order"] == list(range(10))
    assert panel["indices_by_label"]["3"] == 55008

    contract["sample_panel"]["indices_by_label"]["3"] = 55009
    with pytest.raises(RuntimeError, match="BLOCKED_SAMPLE_CONTRACT_CHANGED"):
        renderer._fixed_panel(contract)


def test_mmf_family_coefficients_preserve_actual_flat_ordering(renderer) -> None:
    class FakeMMF:
        metadata = {
            "mode_family_names": [
                name
                for port in range(100)
                for name in (("LP01", "LP11a") if port >= 0 else ())
            ]
            + [
                family
                for family in renderer.MMF_SYNTHETIC_EXTRA_FAMILIES
                for _port in range(100)
            ],
            "ordering_contract": "exact_fmf_prefix_interleaved_per_port_then_additional_candidate_blocks_y_then_x",
        }

    flat = torch.ones((2, 600), dtype=torch.complex64)
    family = torch.ones((2, 6, 100), dtype=torch.complex64)
    result = renderer._family_coefficients(
        "MMF-M6",
        FakeMMF(),
        {"coefficients": flat, "mode_family_coefficients": family},
    )
    _flat, family_out, names, flat_names, ordering = result
    assert tuple(family_out.shape) == (2, 6, 100)
    assert names == list(renderer.MMF_FAMILY_ORDER)
    assert flat_names[:4] == ["LP01", "LP11a", "LP01", "LP11a"]
    assert flat_names[200:300] == ["ETA1"] * 100
    assert flat_names[500:600] == ["ETA2"] * 100
    assert "exact_fmf_prefix_interleaved" in ordering["ordering_contract"]


def test_fmf_family_coefficients_make_family_axis_explicit(renderer) -> None:
    flat = torch.ones((2, 200), dtype=torch.complex64)
    port_mode = torch.ones((2, 100, 2), dtype=torch.complex64)

    class FakeFMF:
        mode_names = ("LP01", "LP11a")

    _flat, family, names, flat_names, _ordering = renderer._family_coefficients(
        "FMF",
        FakeFMF(),
        {"port_output_coefficients": flat, "mode_coefficients": port_mode},
    )
    assert tuple(family.shape) == (2, 2, 100)
    assert names == ["LP01", "LP11a"]
    assert flat_names[:4] == ["LP01", "LP11a", "LP01", "LP11a"]


def test_phase_view_uses_raw_angle_and_registered_mask(renderer) -> None:
    value = torch.ones((1, 1, 4, 4), dtype=torch.complex64)
    value[0, 0, 0, 0] = 0.001 + 0.0j
    phase, threshold, valid_fraction = renderer._phase_view(value, 0)
    assert threshold == pytest.approx(0.01)
    assert np.isnan(phase[-1, -1])
    assert valid_fraction == pytest.approx(15 / 16)
    assert np.all(np.abs(phase[np.isfinite(phase)]) <= np.pi)


def test_detector_delta_is_descriptive_and_has_no_alignment(renderer) -> None:
    native = torch.arange(512, dtype=torch.float32).reshape(2, 256)
    result = renderer._detector_deltas(
        {"SMF": native, "FMF": native + 1.0, "MMF-M6": native + 2.0}
    )
    assert result["reference_arm"] == "SMF"
    assert result["arms"]["FMF"]["alignment_or_remap"] is False
    assert result["arms"]["MMF-M6"]["refit"] is False
    assert len(result["arms"]["FMF"]["delta_l2"]["per_sample"]) == 2


def test_renderer_has_no_optimizer_or_training_entrypoint(renderer) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "optimizer.step" not in source
    assert "loss.backward" not in source
    assert "generate_from_prior" not in source
    assert "full_mode_banks_stored" in source
