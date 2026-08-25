from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from ldm_project.v2.application.research import joint_quadrature_optical_capacity as jqoc


def _ports(field: torch.Tensor) -> dict[str, torch.Tensor]:
    return jqoc._balanced_ports(field, lo_amplitude=0.7, exposure=0.5)


def _row(dimension: int, seed: int, *, electronic: bool = True, intensity: bool = False, quadrature: bool = True, qsafe: bool = True, intensity_semantic: bool = True, intensity_agreement: bool = True) -> dict:
    return {
        "dimension": dimension,
        "seed": seed,
        "count": 1000,
        "finite": True,
        "electronic_pass": electronic,
        "intensity_pass": intensity,
        "quadrature_pass": quadrature,
        "quadrature_safety_pass": qsafe,
        "intensity": {"semantic_pass": intensity_semantic, "agreement_pass": intensity_agreement},
        "electronic": {},
        "quadrature": {"packed_shape": [1, 1, 400, 400]},
    }


def _classification_result(rows: list[dict]) -> dict:
    return {
        "banks": {f"{row['seed']}:{row['dimension']}": row for row in rows},
        "measurement_valid": True,
        "optimizer_created": False,
        "checkpoint_created": False,
        "persistent_full_field_saved": False,
        "parameter_hash_before": {"system": "same"},
        "parameter_hash_after": {"system": "same"},
        "historical_replay": {"matches": True},
        "chunk_exact": True,
        "controls": {
            "zero_code": {},
            "label_shift": {},
            "lo_off": {},
            "quadrature_identity": {},
            "sign_inversion": {},
            "quadrature_swap": {},
            "packing": {"shape": [10, 1, 400, 400]},
            "repeated_execution_exact": True,
            "jacobian": {str(dimension): {"rank_pass": True, "singular_ratio_pass": True, "finite": True} for dimension in jqoc.CODE_DIMS},
            "passes": True,
            "receiver": {},
        },
    }


def test_config_and_cli_registration() -> None:
    config = jqoc._config("configs/research/joint_quadrature_optical_capacity_v1.yaml")
    assert config["code_dims"] == [8, 16, 32]
    assert config["receiver_sensor_shape"] == [400, 400]


def test_dct_basis_is_nested_signed_and_finite() -> None:
    basis = jqoc._dct_basis()
    assert basis.shape == (32, 28, 28)
    assert torch.isfinite(basis).all()
    assert torch.allclose(basis.square().mean(dim=(1, 2)), torch.ones(32), atol=1e-5)
    assert torch.equal(basis, jqoc._dct_basis())
    assert torch.all(basis.flatten(1).gather(1, basis.abs().flatten(1).argmax(1, keepdim=True)).squeeze(1) >= 0)


def test_large_calibration_quantile_uses_registered_linear_definition() -> None:
    values = torch.arange(11, dtype=torch.float32)
    assert torch.equal(jqoc._linear_quantile(values, 0.0), torch.tensor(0.0))
    assert torch.equal(jqoc._linear_quantile(values, 1.0), torch.tensor(10.0))
    assert torch.equal(jqoc._linear_quantile(values, 0.95), torch.tensor(9.5))


def test_four_port_identity_and_physical_packing() -> None:
    generator = torch.Generator(device="cpu").manual_seed(3)
    field = torch.complex(torch.randn((2, 1, 200, 200), generator=generator), torch.randn((2, 1, 200, 200), generator=generator))
    values = _ports(field)
    assert torch.isfinite(torch.stack(tuple(values.values()))).all()
    assert values["r_plus"].shape == (2, 1, 200, 200)
    assert values["j_plus"].shape == (2, 1, 200, 200)
    assert values["r_energy_residual"].abs().max() < 1e-5
    assert values["j_energy_residual"].abs().max() < 1e-5
    packed = jqoc._pack_ports({name: values[name] for name in ("r_plus", "r_minus", "j_plus", "j_minus")})
    assert packed.shape == (2, 1, 400, 400)


def test_classification_is_dimension_grouped_and_fail_closed() -> None:
    rows = [_row(d, seed) for d in jqoc.CODE_DIMS for seed in jqoc.BANK_SEEDS]
    result = _classification_result(rows)
    assert jqoc._classify(result)[0] == "quadrature_observability_rescue_admitted"
    rows = [_row(d, seed, intensity=d == 16, quadrature=d == 32) for d in jqoc.CODE_DIMS for seed in jqoc.BANK_SEEDS]
    assert jqoc._classify(_classification_result(rows)) == ("intensity_observability_sufficient_on_structured_canary", "register_16d_structured_intensity_architecture_preflight")
    rows = [_row(d, seed, electronic=False) for d in jqoc.CODE_DIMS for seed in jqoc.BANK_SEEDS]
    assert jqoc._classify(_classification_result(rows))[0] == "electronic_structured_input_capacity_failed"
    rows = [_row(d, seed, qsafe=False, quadrature=False) for d in jqoc.CODE_DIMS for seed in jqoc.BANK_SEEDS]
    assert jqoc._classify(_classification_result(rows))[0] == "quadrature_receiver_safety_failed"
    rows = [_row(d, seed, intensity_semantic=False) for d in jqoc.CODE_DIMS for seed in jqoc.BANK_SEEDS]
    assert jqoc._classify(_classification_result(rows))[0] == "optical_semantic_carrier_failed"
    rows = [_row(d, seed, intensity_semantic=True, intensity_agreement=False) for d in jqoc.CODE_DIMS for seed in jqoc.BANK_SEEDS]
    assert jqoc._classify(_classification_result(rows))[0] == "generated_evaluator_consensus_failed"
    rows = [_row(d, seed, quadrature=False) for d in jqoc.CODE_DIMS for seed in jqoc.BANK_SEEDS]
    assert jqoc._classify(_classification_result(rows))[0] == "quadrature_code_observability_failed"
    invalid = _classification_result([_row(8, jqoc.BANK_SEEDS[0])])
    invalid["historical_replay"]["matches"] = False
    assert jqoc._classify(invalid)[0] == "invalid_harness"
    missing = _classification_result(rows)
    del missing["banks"]["57721:8"]
    assert jqoc._classify(missing)[0] == "invalid_harness"
    nonfinite = _classification_result(rows)
    nonfinite["banks"]["57721:8"]["finite"] = False
    assert jqoc._classify(nonfinite)[0] == "invalid_harness"


def test_manifest_only_dispatch_rejects_legacy_invocation() -> None:
    with pytest.raises(ValueError, match="manifest-only"):
        jqoc.dispatch(Namespace(research_action="audit", run_manifest=None))


@pytest.mark.parametrize("action", ["report", "closeout"])
def test_consumers_do_not_initialize_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("schema: test\n", encoding="utf-8")
    source_stage = "audit" if action == "report" else "report"
    evidence = tmp_path / "source.json"
    evidence.write_text(json.dumps({"schema": f"{jqoc.SCHEMA_PREFIX}_{source_stage}.v1", "status": "valid", "invocation_mode": "manifest", "legacy_diagnostic_only": False, "permitted_consumers": [action], "passed": False, "measurement_valid": True, "classification": "quadrature_code_observability_failed", "next_action": "close_joint_quadrature_capacity_route"}), encoding="utf-8")
    output = tmp_path / f"{action}.json"
    monkeypatch.setattr(jqoc, "prepare_args", lambda namespace, **_: namespace)
    monkeypatch.setattr(jqoc, "validate_manifest_namespace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(jqoc, "_stage", lambda *_args, **_kwargs: {"action": "RUN"})
    monkeypatch.setattr(jqoc, "_require_cuda", lambda *_args: pytest.fail("pure consumer initialized CUDA"))
    args = Namespace(research_action=action, run_manifest=str(manifest), config="configs/research/joint_quadrature_optical_capacity_v1.yaml", dataset_root="data", output=str(output), measurement_evidence=str(evidence))
    assert jqoc.dispatch(args) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["stage"] == action
