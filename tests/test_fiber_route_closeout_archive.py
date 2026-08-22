import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
ARCHIVE = ROOT / "experiments/archive/v2_fiber_route_no_e1_closeout"
EVIDENCE = ARCHIVE / "evidence"


def _json(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_index_matches_every_evidence_file() -> None:
    index = json.loads((ARCHIVE / "archive_index.json").read_text())
    assert index["status"] == "closed_synthetic_fiber_route_no_e1"
    assert index["diagnostic_only"] is True
    assert index["runnable"] is False
    assert len(index["files"]) == 17
    for entry in index["files"]:
        path = ROOT / entry["archive_path"]
        assert path.is_file()
        assert path.stat().st_size == entry["bytes"]
        assert _sha(path) == entry["sha256"]
        assert not path.name.endswith((".pt", ".pth", ".ckpt"))


def test_closeout_states_and_classifications_are_exact() -> None:
    v3 = _json("v3_closeout.json")
    interface = _json("interface_closeout.json")
    loss = _json("loss_exposure_closeout.json")
    decoder = _json("decoder_closeout.json")
    assert (v3["gate"], v3["conclusion"]) == (
        "valid_failed",
        "no_reproducible_fiber_gain",
    )
    assert (interface["gate"], interface["conclusion"]) == (
        "valid",
        "class_encoding_retained_semantic_manifold_mismatch",
    )
    assert (loss["gate"], loss["conclusion"]) == (
        "valid",
        "global_exposure_explains_oracle_only",
    )
    assert (decoder["gate"], decoder["conclusion"]) == (
        "valid",
        "zero_phase_decoder_local_reachability_failed",
    )
    assert decoder["next_action"] == "close_current_fiber_route_no_e1"
    for closeout in (v3, interface, loss, decoder):
        assert closeout["formal_promotion"] is False
        assert closeout["synthetic_diagnostic_only"] is True
    for closeout in (interface, loss, decoder):
        assert closeout["training_authorized"] is False


def test_cross_stage_hash_bindings_are_preserved() -> None:
    hashes = {
        name: _sha(EVIDENCE / name)
        for name in (p.name for p in EVIDENCE.glob("*.json"))
    }
    assert _json("v3_closeout.json")["selection_sha256"] == hashes["v3_selection.json"]
    interface = _json("interface_closeout.json")
    assert interface["v3_closeout_sha256"] == hashes["v3_closeout.json"]
    assert interface["attribution_measurement_sha256"] == hashes["attribution_v1_measurement.json"]
    assert interface["validation_sha256"] == hashes["interface_validation.json"]
    assert interface["measurement_sha256"] == hashes["interface_measurement.json"]
    assert interface["presentation_sha256"] == hashes["interface_presentation.json"]
    loss = _json("loss_exposure_closeout.json")
    assert loss["interface_closeout_sha256"] == hashes["interface_closeout.json"]
    assert loss["interface_measurement_sha256"] == hashes["interface_measurement.json"]
    assert loss["validation_sha256"] == hashes["loss_exposure_validation.json"]
    assert loss["calibration_sha256"] == hashes["loss_exposure_calibration.json"]
    assert loss["measurement_sha256"] == hashes["loss_exposure_measurement.json"]
    assert loss["presentation_sha256"] == hashes["loss_exposure_presentation.json"]
    decoder = _json("decoder_closeout.json")
    assert decoder["validation_sha256"] == hashes["decoder_validation.json"]
    assert decoder["design_sha256"] == hashes["decoder_design.json"]
    assert decoder["measurement_sha256"] == hashes["decoder_measurement.json"]
    assert decoder["presentation_sha256"] == hashes["decoder_presentation.json"]
    validation = _json("decoder_validation.json")
    assert validation["interface_closeout_sha256"] == hashes["interface_closeout.json"]
    assert validation["loss_exposure_closeout_sha256"] == hashes["loss_exposure_closeout.json"]
    assert validation["loss_exposure_measurement_sha256"] == hashes["loss_exposure_measurement.json"]


def test_summary_and_manifest_close_the_route() -> None:
    summary = json.loads((ARCHIVE / "fiber_route_closeout_summary.json").read_text())
    assert summary["status"] == "closed_synthetic_fiber_route_no_e1"
    assert summary["formal_promotion"] is False
    assert summary["training_authorized"] is False
    assert summary["synthetic_diagnostic_only"] is True
    assert summary["rejection_decision"] == [
        "no_fiber_training",
        "no_e1",
        "no_modal_regularizer",
        "no_formal_promotion",
    ]
    assert summary["prohibited_claims"] == [
        "real-fiber performance failure",
        "phase-only capacity failure",
        "modal-capacity failure",
        "generator class encoding absent",
        "universal evaluator incompatibility",
    ]
    assert len(summary["restart_contract"]) == 4
    manifest = yaml.safe_load((ROOT / "experiments/archive/manifest.yaml").read_text())
    entries = [item for item in manifest["archives"] if item["id"] == "v2-fiber-route-no-e1-closeout"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["diagnostic_only"] is True
    assert entry["runnable"] is False
    assert entry["archived_path"] == "experiments/archive/v2_fiber_route_no_e1_closeout/"
    assert "experiments/archive/v2_fiber_route_no_e1_closeout/fiber_route_closeout_summary.json" in entry["evidence_paths"]


def test_capsule_contains_no_runnable_or_large_artifact_inputs() -> None:
    forbidden_parts = {"failed-runs", "checkpoints", "dataset"}
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".png", ".jpg", ".jpeg", ".npy", ".npz"}
    for path in ARCHIVE.rglob("*"):
        if not path.is_file():
            continue
        assert not any(part in forbidden_parts for part in path.parts)
        assert path.suffix not in forbidden_suffixes
    assert "python -m ldm_project.v2" not in (ARCHIVE / "README.md").read_text()
