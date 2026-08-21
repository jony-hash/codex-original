from __future__ import annotations

from types import SimpleNamespace

import torch

from ldm_project.v2.application.cli import _parser
from ldm_project.v2.application.research import fiber_interface_attribution as interface
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec


def _lineage_args(action: str) -> list[str]:
    values = [
        "research",
        "fiber-interface-attribution",
        action,
    ]
    if action in {"validate", "run"}:
        values.extend(
            [
                "--config",
                "config.yaml",
                "--dataset-root",
                "data",
                "--primary-evaluator-checkpoint",
                "primary.pt",
                "--secondary-evaluator-checkpoint",
                "secondary.pt",
                "--primary-qualification-evidence",
                "primary.json",
                "--secondary-qualification-evidence",
                "secondary.json",
                "--research-initialization-checkpoint",
                "init.pt",
                "--source-training-evidence",
                "source.json",
                "--v3-closeout",
                "v3.json",
                "--attribution-measurement",
                "attribution.json",
            ]
        )
    if action == "validate":
        values.extend(["--lp01-fixture", "fixture.json"])
    if action == "run":
        values.extend(["--lp01-fixture", "fixture.json", "--validation-evidence", "validation.json"])
    if action == "report":
        values.extend(["--measurement-evidence", "measurement.json"])
    if action == "closeout":
        values.extend(["--measurement-evidence", "measurement.json", "--presentation-evidence", "presentation.json"])
    values.extend(["--output", "out.json"])
    return values


def test_interface_cli_has_four_independent_lifecycle_actions() -> None:
    parser = _parser()
    for action in ("validate", "run", "report", "closeout"):
        parsed = parser.parse_args(_lineage_args(action))
        assert parsed.research_program == "fiber-interface-attribution"
        assert parsed.research_action == action


def test_projection_residual_is_orthogonal_and_r100_is_identity() -> None:
    plane = PlaneSpec("encoder_input", (100, 100), (2.0e-6, 2.0e-6), 532.0e-9, 0.1, "encoder")
    generator = torch.Generator().manual_seed(17)
    values = torch.complex(
        torch.randn((2, 1, 100, 100), generator=generator),
        torch.randn((2, 1, 100, 100), generator=generator),
    )
    field = ComplexField(values, plane, OperationProvenance("test_input", None, None))
    r100 = interface._ResidualBlendOLS(plane, alpha=1.0)(field)
    assert torch.allclose(r100.tensor, values, atol=1.0e-5, rtol=1.0e-5)
    operator = interface._ResidualBlendOLS(plane, alpha=0.0)
    operator(field)
    info = operator.last_diagnostics
    closure = info["input_energy"] - info["projected_energy"] - info["residual_energy"]
    assert float((closure.abs() / info["input_energy"].clamp_min(1.0e-30)).max()) <= 1.0e-5
    assert info["passive"] is True


def test_rank_summary_and_shuffled_centroid_are_finite() -> None:
    coefficients = torch.complex(torch.randn(30, 100, generator=torch.Generator().manual_seed(1)), torch.randn(30, 100, generator=torch.Generator().manual_seed(2)))
    summary = interface._rank_summary(coefficients)
    assert summary["rank_at_1e-8"] <= 29
    assert torch.isfinite(torch.tensor(summary["singular_values"])).all()
    labels = torch.arange(10).repeat_interleave(3)
    embeddings = {"S0": {"ols_output": {panel: torch.randn(30, 4, generator=torch.Generator().manual_seed(panel)) for panel in interface.PANELS}}}
    result = interface._cross_panel_nearest(embeddings, {panel: labels for panel in interface.PANELS})
    assert 0.0 <= result["S0"]["ols_output"]["shuffled_raw"] <= 1.0


def test_energy_oracle_is_explicitly_nonphysical() -> None:
    assert "S0-energy-oracle" not in interface.PHYSICAL_VARIANTS
    assert interface.VARIANTS[-1] == "S0-energy-oracle"
    bands = interface._residual_spectrum_bands(
        torch.ones(2, 100, 100, dtype=torch.complex64)
    )
    assert torch.allclose(bands.sum(1), torch.ones(2), atol=1.0e-5)


def test_classification_has_four_fail_closed_outcomes() -> None:
    variants = list(interface.VARIANTS)

    def payload(*, gain: float, retained: bool, collapsed: bool, low_rank: bool) -> dict:
        rows = {variant: {} for variant in variants}
        for variant in variants:
            for panel in interface.PANELS:
                rows[variant][str(panel)] = {
                    "secondary_accuracy": 0.1 + (gain if variant == "R25" else 0.0),
                    "secondary_min_recall": 0.1 if variant == "R25" and gain > 0 else 0.0,
                    "secondary_prediction_histogram": [0] * 10,
                    "port_rank": {"rank_at_1e-8": 7 if low_rank else 8},
                }
                rows[variant][str(panel)]["secondary_prediction_histogram"][8] = 500 if collapsed else 50
        nearest_value = 0.95 if retained else 0.5
        nearest = {variant: {stage: {"raw": nearest_value, "unit": nearest_value, "shuffled_raw": 0.1, "shuffled_unit": 0.1} for stage in interface.STAGES} for variant in variants}
        return {
            "rows": rows,
            "nearest_centroid": nearest,
            "jacobian": {
                "points": [
                    {"port_rank_1e-3": 7 if low_rank else 8, "cmos_rank_1e-3": 7 if low_rank else 8}
                    for _ in range(30)
                ]
            },
            "bootstrap": {variant: {"observed_gain": gain if variant == "R25" else 0.0, "ci95": [0.001 if gain > 0 else -0.01, 0.03], "replicates": 10000} for variant in ("R25", "R50", "R75", "R100")},
            "energy_oracle_bootstrap": {
                "S0-energy-oracle": {
                    "observed_gain": 0.02 if collapsed else 0.0,
                    "ci95": [0.01 if collapsed else -0.01, 0.03],
                }
            },
        }

    assert interface._classification(payload(gain=0.02, retained=False, collapsed=False, low_rank=False))["conclusion"] == "modal_interface_candidate_admitted"
    assert interface._classification(payload(gain=0.0, retained=True, collapsed=True, low_rank=False))["conclusion"] == "class_encoding_retained_semantic_manifold_mismatch"
    assert interface._classification(payload(gain=0.0, retained=False, collapsed=False, low_rank=True))["conclusion"] == "port_observability_limited"
    assert interface._classification(payload(gain=0.0, retained=False, collapsed=False, low_rank=False))["conclusion"] == "interface_inconclusive"
    oracle = interface._classification(payload(gain=0.0, retained=False, collapsed=True, low_rank=False))
    assert oracle["energy_oracle_only"] is True


def test_dirty_worktree_rejects_run_creation(tmp_path, monkeypatch) -> None:
    output = tmp_path / "validation.json"
    monkeypatch.setattr(interface, "_dirty", lambda: True)
    result = interface.validate(SimpleNamespace(output=str(output)))
    assert result == 2
    payload = output.read_text(encoding="utf-8")
    assert '"gate": "invalid_harness"' in payload


def test_tampered_measurement_tensor_is_rejected(tmp_path) -> None:
    root = tmp_path / "measurement"
    rows = {variant: {} for variant in interface.VARIANTS}
    pairing = {"labels": {variant: {} for variant in interface.VARIANTS}, "codes": {variant: {} for variant in interface.VARIANTS}}
    for variant in interface.VARIANTS:
        for panel in interface.PANELS:
            values = {
                "labels": torch.arange(10),
                "codes": torch.zeros(10, 8),
                "fixed_cmos": torch.zeros(10, 1, 28, 28),
                "primary_prediction": torch.zeros(10, dtype=torch.long),
                "secondary_prediction": torch.zeros(10, dtype=torch.long),
                "secondary_correctness": torch.ones(10),
                "representative_trace": [],
            }
            if variant in interface.PHYSICAL_VARIANTS or variant == "S0-energy-oracle":
                values["port_input_coefficients"] = torch.zeros(10, 100, dtype=torch.complex64)
                values["port_output_coefficients"] = torch.zeros(10, 100, dtype=torch.complex64)
            refs = interface._save_variant_artifacts(root, variant, panel, values)
            rows[variant][str(panel)] = {"artifacts": refs}
            pairing["labels"][variant][str(panel)] = interface._tensor_hash(values["labels"])
            pairing["codes"][variant][str(panel)] = interface._tensor_hash(values["codes"])
    payload = {"variants": list(interface.VARIANTS), "panels": list(interface.PANELS), "rows": rows, "pairing": pairing}
    interface._validate_measurement_artifacts(root, payload)
    tampered = root / "measurements" / "I0" / str(interface.PANELS[0]) / "labels.pt"
    interface.atomic_torch_save(tampered, torch.ones(10))
    try:
        interface._validate_measurement_artifacts(root, payload)
    except ValueError as error:
        assert "artifact" in str(error) or "hash" in str(error)
    else:
        raise AssertionError("tampered measurement tensor was accepted")


def test_stale_validation_is_rejected_before_run_directory_creation(tmp_path) -> None:
    validation = tmp_path / "validation.json"
    validation.write_text('{"schema": "fiber_interface_validation.v1", "gate": "invalid_harness"}\n', encoding="utf-8")
    output = tmp_path / "run"
    args = SimpleNamespace(validation_evidence=str(validation), output=str(output))
    try:
        interface.run(args)
    except ValueError as error:
        assert "validation" in str(error)
    else:
        raise AssertionError("invalid validation was accepted")
    assert not output.exists()
