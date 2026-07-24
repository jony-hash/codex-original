"""Export diagnostic phase maps only after an explicit held-out acceptance gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


NAMES = ("encoder_l1", "encoder_l2", "decoder_l1", "decoder_l2")


def export(*, report: str | Path, phases: str | Path, output: str | Path) -> dict[str, object]:
    gate = json.loads(Path(report).read_text(encoding="utf-8"))
    required = ("corr", "dice", "saturation_error", "prototype_label_accuracy", "camera_code_factor")
    metrics = gate.get("held_out_test", gate.get("metrics", {}))
    if gate.get("manual_readability") is not True or any(name not in metrics for name in required):
        raise RuntimeError("v67 phase export requires explicit held-out metrics and manual_readability=true")
    if not (
        metrics["corr"] >= 0.78 and metrics["dice"] >= 0.72
        and metrics["saturation_error"] <= 0.08
        and metrics["prototype_label_accuracy"] >= 0.80
        and metrics["camera_code_factor"] >= 0.25
    ):
        raise RuntimeError("v67 held-out acceptance thresholds are not met")
    source = Path(phases)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for name in NAMES:
        phase = np.asarray(np.load(source / f"{name}_phase_rad.npy"), dtype=np.float32)
        if phase.shape != (100, 100):
            raise ValueError(f"{name} must be 100x100, got {phase.shape}")
        phase = np.mod(phase, 2.0 * np.pi)
        target = destination / f"{name}_phase_rad.npy"
        np.save(target, phase)
        files[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        "diagnostic_only": True,
        "fabrication_status": "ideal-phase-fabrication-candidate",
        "height_map_exported": False,
        "grid_shape": [100, 100],
        "pixel_pitch_m": 2e-6,
        "active_fov_m": 200e-6,
        "phase_range_rad": [0.0, float(2.0 * np.pi)],
        "installation": "all_phase_structures_face_encoder_incident_direction",
        "files_sha256": files,
        "height_calibration": "requires measured n(532nm) and height-phase calibration",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--phases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    print(json.dumps(export(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
