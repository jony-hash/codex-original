from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "validate_v58_2_relay_reference.py"


def test_v58_2_report_is_layered_and_excludes_invalid_full_size_rs():
    spec = importlib.util.spec_from_file_location("v58_2_relay", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run(ROOT / "configs" / "validate_v55_coherent_relay.yaml", "cpu")
    assert report["validation_revision"] == "v58.2-layered-b0-relay-validation"
    assert report["status"] == "physical-forward-model-valid"
    assert report["production_analytic_validation"]["periodic_boundary_condition"] is True
    assert report["pupil_validation"]["pass_stop_response_error"] <= 0.01
    assert report["scaled_system_propagation_validation"]["boundary_energy_fraction"] <= 1e-4
    assert "excluded from acceptance" in report["sampling_audit"]["actual_size_low_resolution_direct_rs"]
