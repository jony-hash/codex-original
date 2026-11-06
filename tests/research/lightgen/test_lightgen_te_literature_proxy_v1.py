from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = REPO_ROOT / "scripts/research/lightgen/diagnose_lightgen_te_literature_proxy_v1.py"
CONFIG = REPO_ROOT / "configs/research/lightgen/te_literature_proxy_v1.yaml"


def _run(mode: str, config_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT), mode, "--config", str(config_path), "--output-dir", str(output_dir)],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )


def test_preflight_and_measurement_consume_saved_two_state_sweep(tmp_path: Path) -> None:
    output_dir = tmp_path / "diagnostic-only_attempt_01"
    config_copy = tmp_path / "research_config.yaml"
    config_copy.write_bytes(CONFIG.read_bytes())

    preflight = _run("preflight", config_copy, output_dir)
    assert preflight.returncode == 0, preflight.stderr
    receipt = json.loads((output_dir / "preflight.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert all(check["passed"] for check in receipt["checks"])

    measurement = _run("measure", config_copy, output_dir)
    assert measurement.returncode == 0, measurement.stderr
    result = json.loads((output_dir / "measurement.json").read_text(encoding="utf-8"))
    assert result["status"] == "valid"
    assert result["model_identity"]["sample_identity"] == "LIT-TE-ADMA2025-THERMOCHROMIC-FIGURE-PROXY"
    assert result["model_identity"]["reference_plane_nm"] == 50.0

    with (output_dir / "thickness_sweep.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 100
    assert {float(row["thickness_nm"]) for row in rows} == set(range(1, 51))
    assert {row["state"] for row in rows} == {"solid_25C", "quasi_liquid_80C"}
    for row in rows:
        reflected = float(row["R"])
        transmitted = float(row["T"])
        absorbed = float(row["A_flux"])
        tau = complex(float(row["tauE_real"]), float(row["tauE_imag"]))
        r_field = complex(float(row["rE_real"]), float(row["rE_imag"]))
        assert abs(reflected - abs(r_field) ** 2) < 1e-12
        assert abs(transmitted - abs(tau) ** 2) < 1e-12
        assert abs(reflected + transmitted + absorbed - 1.0) < 1e-12

    assert (output_dir / "transmission_vs_thickness.pdf").is_file()
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "0.0053674261" in report
    assert "not optical-intensity nonlinearity" in report


def test_measure_rejects_preflight_for_a_different_sample_identity(tmp_path: Path) -> None:
    output_dir = tmp_path / "diagnostic-only_attempt_02"
    config_path = tmp_path / "research_config.yaml"
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["material"]["sample_identity"] = "different-proxy-identity"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    preflight = _run("preflight", config_path, output_dir)
    assert preflight.returncode == 0, preflight.stderr
    config["material"]["sample_identity"] = "changed-after-preflight"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    measure = _run("measure", config_path, output_dir)
    assert measure.returncode == 2
    assert "Preflight does not pass" in measure.stderr
    assert not (output_dir / "measurement.json").exists()
