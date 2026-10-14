from __future__ import annotations

import csv
import math

import pytest
import torch

from scripts.research.lightgen import (
    audit_lightgen_2d_generator_cross_port_joint_structure_v1 as audit,
)


def _population() -> tuple[torch.Tensor, list[str]]:
    labels = torch.tensor([digit for _ in range(50) for digit in range(10)], dtype=torch.long)
    roles = ["fit"] * audit.FIT_ROWS + ["confirmation"] * audit.CONFIRM_ROWS
    return labels, roles


def _toy_optical_raw(z: torch.Tensor) -> torch.Tensor:
    mode = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex128) / math.sqrt(2.0)
    amplitude = z[:, :2].to(torch.complex128) @ mode.T
    intensity = amplitude.abs().square().to(torch.float64)
    raw = torch.zeros((z.shape[0], 1, 200, 200), dtype=torch.float64)
    raw[:, :, :100, :] = intensity[:, 0, None, None, None]
    raw[:, :, 100:, :] = intensity[:, 1, None, None, None]
    return raw


def _analytic_joint_population() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    labels, roles = _population()
    z = torch.ones((audit.TOTAL_ROWS, audit.PORTS), dtype=torch.complex64)
    confirm_zero = ((labels == 0) & (torch.arange(audit.TOTAL_ROWS) >= audit.FIT_ROWS)).nonzero().flatten()
    z[confirm_zero[15:], :2] = -1.0 + 0.0j
    targets = _toy_optical_raw(z)
    return z, targets, labels, roles


def _synthetic_class_rows(z: torch.Tensor, targets: torch.Tensor, labels: torch.Tensor, roles: list[str]) -> list[dict[str, object]]:
    maps = audit._scenario_mappings(labels, roles)
    rows = []
    seed = audit.SHUFFLE_SEEDS[0]
    observed = {
        "complete_joint": (z, None),
        "whole_row_control": (z.gather(0, maps["whole_row_control"]), audit.WHOLE_ROW_SEED),
        f"port_shuffle_{seed}": (z.gather(0, maps[f"port_shuffle_{seed}"]), seed),
    }
    seed_rows = []
    for scenario, (scenario_z, scenario_seed) in observed.items():
        raw = _toy_optical_raw(scenario_z)
        scenario_rows = audit._class_distances(raw, targets, labels, roles, scenario=scenario, seed=scenario_seed, shape_size=28)
        rows.extend(scenario_rows)
        if scenario_seed == seed:
            seed_rows = scenario_rows
    for other_seed in audit.SHUFFLE_SEEDS[1:]:
        rows.extend({**row, "scenario": f"port_shuffle_{other_seed}", "seed": other_seed} for row in seed_rows)
    return rows


def test_production_mapping_preserves_each_complex_port_marginal_and_role() -> None:
    labels, roles = _population()
    z = torch.arange(audit.TOTAL_ROWS * audit.PORTS, dtype=torch.float32).reshape(audit.TOTAL_ROWS, audit.PORTS)
    source = torch.complex(z, -z)
    mapping = audit._permutation_map(labels, roles, seed=audit.SHUFFLE_SEEDS[0], portwise=True)
    changed = audit._apply_map(source, mapping, labels, roles)

    for digit in audit.CLASSES:
        for role, bounds in (("fit", (0, audit.FIT_ROWS)), ("confirmation", (audit.FIT_ROWS, audit.TOTAL_ROWS))):
            start, stop = bounds
            carriers = ((labels[start:stop] == digit).nonzero().flatten() + start)
            for port in range(audit.PORTS):
                donors = mapping[carriers, port]
                assert torch.unique(donors).numel() == carriers.numel()
                assert torch.equal(changed[carriers, port], source[donors, port])
                assert torch.equal(torch.sort(changed[carriers, port].real).values, torch.sort(source[carriers, port].real).values)
                assert torch.equal(torch.sort(changed[carriers, port].imag).values, torch.sort(source[carriers, port].imag).values)


def test_production_forward_uses_single_row_microbatches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_infer_batch(
        runtime: torch.nn.Module,
        evaluator: torch.nn.Module,
        z: torch.Tensor,
        device: torch.device,
        camera_pixel_area: float,
    ) -> dict[str, torch.Tensor]:
        del runtime, evaluator, device, camera_pixel_area
        calls.append(z.shape[0])
        return {"readout": z[:, :1].real.to(torch.float32)}

    monkeypatch.setattr(audit, "_infer_batch", fake_infer_batch)
    z = torch.ones((3, audit.PORTS), dtype=torch.complex64)
    result = audit._infer_rows(torch.nn.Identity(), torch.nn.Identity(), z, torch.device("cpu"), 1.0)

    assert calls == [1, 1, 1]
    assert result["readout"].shape == (3, 1)
    monkeypatch.undo()
    with pytest.raises(audit.JointStructureError, match="single complex OLS row"):
        audit._infer_batch(torch.nn.Identity(), torch.nn.Identity(), z, torch.device("cpu"), 1.0)


def test_production_shape_measurement_and_saved_csv_detect_joint_pairing(tmp_path) -> None:
    z, targets, labels, roles = _analytic_joint_population()
    rows = _synthetic_class_rows(z, targets, labels, roles)
    decision = audit._primary_decision(rows)

    assert torch.equal(z.abs().square().sum(1), z.gather(0, audit._scenario_mappings(labels, roles)[f"port_shuffle_{audit.SHUFFLE_SEEDS[0]}" ]).abs().square().sum(1))
    assert decision["complete_joint_mean"] == pytest.approx(0.0, abs=1e-12)
    assert decision["difference_shuffled_minus_joint"] > decision["tolerance"]
    assert decision["classification"] == "JOINT_PAIRING_SHAPE_SUPPORT"

    population = {"source_row_indices": list(range(audit.TOTAL_ROWS)), "labels": labels.tolist(), "roles": roles}
    maps = audit._scenario_mappings(labels, roles)
    paired_rows = []
    for scenario in audit.FORMAL_SCENARIOS:
        seed = audit.WHOLE_ROW_SEED if scenario == "whole_row_control" else ""
        if scenario.startswith("port_shuffle_"):
            seed = int(scenario.removeprefix("port_shuffle_"))
        for position in range(audit.TOTAL_ROWS):
            paired_rows.append({
                "scenario": scenario,
                "seed": seed,
                "position": position,
                "source_row_index": position,
                "label": int(labels[position]),
                "role": roles[position],
                "donor_source_row_indices": json_list(maps[scenario][position].tolist()),
                "generator_input_power": 1.0,
                "camera_integrated_intensity": 1.0,
                "raw_clip_fraction": 0.0,
                "e_d_prediction": int(labels[position]),
                "e_d_correct": 1,
            })
    paired_path = tmp_path / "paired_rows.csv"
    class_path = tmp_path / "class_distances.csv"
    audit._write_csv(paired_path, paired_rows)
    audit._write_csv(class_path, rows)
    replay = audit._recheck_csv(paired_path, population, class_path, decision)
    assert replay["passed"] is True
    assert replay["primary_decision"]["classification"] == "JOINT_PAIRING_SHAPE_SUPPORT"


def test_whole_row_control_leaves_class_shape_sets_unchanged() -> None:
    z, targets, labels, roles = _analytic_joint_population()
    maps = audit._scenario_mappings(labels, roles)
    joint = _toy_optical_raw(z)
    control = _toy_optical_raw(z.gather(0, maps["whole_row_control"]))
    joint_shape = audit._shape_vectors(joint[200:], 28)
    control_shape = audit._shape_vectors(control[200:], 28)
    for digit in audit.CLASSES:
        selected = labels[200:] == digit
        assert audit._energy_distance(joint_shape[selected], control_shape[selected]) <= 1.0e-12
    assert audit._energy_distance(audit._shape_vectors(targets[200:], 28), joint_shape) == pytest.approx(0.0, abs=1e-12)


def test_normalized_unclipped_shapes_ignore_brightness_and_global_phase() -> None:
    z, _, _, _ = _analytic_joint_population()
    raw = _toy_optical_raw(z[:20])
    scaled = _toy_optical_raw(z[:20] * 2.0)
    phase = torch.polar(torch.ones(20), torch.linspace(0.1, 2.5, 20)).to(torch.complex64)
    phase_raw = _toy_optical_raw(z[:20] * phase[:, None])

    shape = audit._shape_vectors(raw, 28)
    assert torch.allclose(shape, audit._shape_vectors(scaled, 28), rtol=0.0, atol=1e-14)
    assert torch.allclose(shape, audit._shape_vectors(phase_raw, 28), rtol=0.0, atol=1e-14)


def test_distance_detects_shape_collapse_when_mean_image_is_preserved() -> None:
    targets = torch.eye(2, dtype=torch.float64)
    collapsed = torch.full((2, 2), 0.5, dtype=torch.float64)

    assert torch.equal(targets.mean(0), collapsed.mean(0))
    assert audit._energy_distance(collapsed, targets) == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-12)


@pytest.mark.parametrize(
    ("joint", "shuffled", "expected"),
    [
        (0.1, 0.1001, "JOINT_PAIRING_SHAPE_SUPPORT"),
        (0.1, 0.0999, "JOINT_PAIRING_NOT_REQUIRED_FOR_THIS_MEASURE"),
        (0.1, 0.10000001, "JOINT_PAIRING_EFFECT_WITHIN_NUMERICAL_TOLERANCE"),
    ],
)
def test_final_decision_distinguishes_support_reverse_and_tolerance(joint: float, shuffled: float, expected: str) -> None:
    result = audit._classify_difference(shuffled - joint, joint, shuffled)
    assert result["classification"] == expected


def test_cross_role_class_reuse_and_missing_port_maps_are_rejected() -> None:
    labels, roles = _population()
    roles_tensor = torch.tensor([0 if role == "fit" else 1 for role in roles], dtype=torch.long)
    mapping = audit._permutation_map(labels, roles, seed=audit.SHUFFLE_SEEDS[1], portwise=True)

    wrong_class = mapping.clone()
    wrong_class[200, 0] = 201
    with pytest.raises(audit.JointStructureError, match="crossed a class boundary"):
        audit._validate_permutation_map(wrong_class, labels, roles_tensor, portwise=True)

    wrong_role = mapping.clone()
    wrong_role[200, 0] = 0
    with pytest.raises(audit.JointStructureError, match="crossed a data role"):
        audit._validate_permutation_map(wrong_role, labels, roles_tensor, portwise=True)

    with pytest.raises(audit.JointStructureError, match=r"int64\[500,100\]"):
        audit._validate_permutation_map(mapping[:, :-1], labels, roles_tensor, portwise=True)


def test_preflight_selects_one_actual_confirmation_row_per_class() -> None:
    labels, roles = _population()
    carriers = audit._preflight_carriers(labels, roles)

    assert carriers.min() >= audit.FIT_ROWS
    assert torch.equal(labels[carriers], torch.arange(10))


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_invalid_shape_inputs_are_rejected(bad: float) -> None:
    raw = torch.ones((1, 1, 200, 200), dtype=torch.float64)
    if bad == 0.0:
        raw.zero_()
    else:
        raw[0, 0, 0, 0] = bad
    with pytest.raises(audit.JointStructureError):
        audit._shape_vectors(raw, 28)


def test_wrong_saved_row_identity_and_view_mismatch_are_rejected() -> None:
    labels, _ = _population()
    population = {"source_row_indices": list(range(audit.TOTAL_ROWS)), "labels": labels.tolist()}
    readout = {
        "source_indices": torch.arange(audit.TOTAL_ROWS),
        "labels": labels.clone(),
        "z": torch.ones((audit.TOTAL_ROWS, audit.PORTS), dtype=torch.complex64),
        "raw_cmos": torch.ones((audit.TOTAL_ROWS, 1, 200, 200), dtype=torch.float32),
        "target_slm_intensity": torch.ones((audit.TOTAL_ROWS, 1, 200, 200), dtype=torch.float32),
        "evaluator_view": torch.ones((audit.TOTAL_ROWS, 1, 28, 28), dtype=torch.float32),
        "e_d_logits": torch.zeros((audit.TOTAL_ROWS, 10), dtype=torch.float32),
    }
    audit._validate_readout(readout, population)

    wrong_ids = dict(readout)
    wrong_ids["source_indices"] = readout["source_indices"].clone()
    wrong_ids["source_indices"][0] += 1
    with pytest.raises(audit.JointStructureError, match="source row IDs differ"):
        audit._validate_readout(wrong_ids, population)

    wrong_view = dict(readout)
    wrong_view["evaluator_view"] = torch.zeros_like(readout["evaluator_view"])
    with pytest.raises(audit.JointStructureError, match="production view differs"):
        audit._validate_readout(wrong_view, population)


def json_list(values: list[int]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"
