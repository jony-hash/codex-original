"""Aggregate verified C2 diagnostic evidence without implying hardware closure."""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--c0-indices", type=Path, required=True)
    p.add_argument("--c0-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.runs
    axis = read(root / "autodl_backup_axis_summary_03/axis_summary.json")["conditions"]
    recheck = read(root / "autodl_backup_axis_recheck_02/recheck.json")["conditions"]
    readout = read(root / "autodl_backup_readout_training_01/results/test_results.json")
    selection = read(root / "autodl_backup_readout_training_01/results/selection.json")
    readout_audit = read(root / "local_readout_audit_01/audit.json")["conditions"]
    pupil = read(root / "autodl_backup_pupil_population_01/results/pupil_population.json")
    pupil_audit = read(root / "local_pupil_audit_01/audit.json")["conditions"]
    refine = read(root / "autodl_backup_pupil_refine_01/refine.json")
    ledger = read(root / "autodl_backup_pupil_ledger_01/ledger.json")
    threshold = read(root / "autodl_backup_threshold_01/threshold.json")
    sensitivity = read(root / "autodl_backup_sensitivity_01/sensitivity.json")
    rotation = read(root / "autodl_backup_rotation_population_01/rotation_population.json")
    grating = read(root / "autodl_backup_pupil_grating_01/pupil_grating.json")
    shapes = read(root / "autodl_backup_shape_02/shape.json")
    margins = read(root / "local_margin_audit_01/margins.json")
    full_pupil = read(root / "autodl_backup_pupil_full_audit_01/audit.json")
    repeat = read(root / "local_pupil_repeat_01/repeat.json")
    indices = read(args.c0_indices)
    c0_data = read(args.c0_data)
    archive_receipt = read(root / "autodl_backup_pupil_full_01/transfer_receipt.json")
    resume = read(root / "autodl_backup_pupil_resume_01/status.json")
    simulated_resume = read(root / "autodl_backup_pupil_resume_sim_01/resume_check.json")
    if (len(full_pupil["conditions"]) != 5 or archive_receipt["status"] != "COMPLETE"
            or archive_receipt["files"] != 443 or resume["state"] != "COMMAND_EXITED_ZERO"
            or simulated_resume["status"] != "PASS" or simulated_resume["rows"] != 32
            or any(v["cpu_prediction_differences"] or v["rows"] != (4157 if k.startswith("test/") else 2475)
                   for k, v in full_pupil["conditions"].items())
            or any(v["rows"] != 16 or v["max_abs_difference"] != 0 for v in repeat["conditions"].values())):
        raise ValueError("Full pupil population verification incomplete")
    if len(margins["conditions"]) != 12:
        raise ValueError("Frozen margin summary is incomplete")
    axis_preflight = read(root / "autodl_backup_axis_summary_03/axis_preflight.json")
    if (axis_preflight["status"] != "READY_AXIS_ALIGNED_ONLY" or axis_preflight["zero_control"] != "PASS"
            or axis_preflight["identity_readout_first_256"] != "PASS"):
        raise ValueError("Axis preflight incomplete")
    suite = ET.parse(root / "test_crossstage_final_03/results.xml").getroot().find("testsuite")
    if suite is None or int(suite.get("errors", "-1")) or int(suite.get("failures", "-1")):
        raise ValueError("Targeted test report is missing or unsuccessful")
    tests, skipped = int(suite.get("tests")), int(suite.get("skipped"))
    focused_suite = ET.parse(root / "test_c2_directed_02/results.xml").getroot().find("testsuite")
    if (focused_suite is None or int(focused_suite.get("tests", "-1")) != 20
            or int(focused_suite.get("errors", "-1")) or int(focused_suite.get("failures", "-1"))
            or int(focused_suite.get("skipped", "-1"))):
        raise ValueError("C2-focused test report is incomplete or unsuccessful")
    focused_tests, focused_skipped = int(focused_suite.get("tests")), int(focused_suite.get("skipped"))
    index_population = {"train": 22279, "validation": 2475, "test": 4157}
    if c0_data.get("identity") != "HW207_MNIST4_C0" or set(c0_data.get("counts", {})) != set(index_population):
        raise ValueError("C0 data manifest identity or roles are invalid")
    seen_indices = {}
    for role, count in index_population.items():
        entry = indices.get(role)
        if entry is None or len(entry.get("original_indices", [])) != count:
            raise ValueError(f"C0 index population mismatch: {role}")
        source = Path(entry["source"]).name
        if source != ("test.pt" if role == "test" else "training.pt"):
            raise ValueError(f"C0 index source mismatch: {role}")
        values = entry["original_indices"]
        if len(set(values)) != count:
            raise ValueError(f"C0 indices contain duplicates: {role}")
        prior = seen_indices.setdefault(source, set())
        if prior.intersection(values):
            raise ValueError(f"C0 data roles overlap: {role}")
        prior.update(values)
        class_counts = c0_data["counts"][role]
        if len(class_counts) != 4 or sum(class_counts) != count:
            raise ValueError(f"C0 class-count manifest mismatch: {role}")
    shape_population = read(root / "autodl_backup_shape_population_03/shape_population.json")
    expected_shape_conditions = {
        f"{count}/{method}/0.92/{role}"
        for count in (26, 28)
        for method in ("nearest", "area")
        for role in ("validation", "test")
    }
    if (shape_population.get("identity") != "FIXED_THRESHOLD_ORIGINAL_GRID_THINNESS_VALIDATION_TEST"
            or set(shape_population.get("conditions", {})) != expected_shape_conditions):
        raise ValueError("Full-population morphology conditions are incomplete")
    shape_audit = {}
    for key, value in shape_population["conditions"].items():
        role = key.split("/")[-1]
        expected_rows = indices[role]["original_indices"]
        records = value.get("per_row", [])
        if value.get("rows") != len(expected_rows) or [item.get("row") for item in records] != expected_rows:
            raise ValueError(f"Morphology row order or population mismatch: {key}")
        for record in records:
            metrics = (record.get("foreground_ratio"), record.get("dice"),
                       record.get("thin_pixel_retention"))
            if (any(v is None or not math.isfinite(float(v)) for v in metrics)
                    or record.get("reference_components") is None
                    or record.get("candidate_components") is None):
                raise ValueError(f"Undefined morphology metric: {key}")
        means = {
            "mean_dice": sum(item["dice"] for item in records) / len(records),
            "mean_thin_pixel_retention": sum(item["thin_pixel_retention"] for item in records) / len(records),
            "mean_component_change": sum(item["candidate_components"] - item["reference_components"]
                                          for item in records) / len(records),
        }
        if any(not math.isclose(value[name], expected_mean, rel_tol=0, abs_tol=1e-12)
               for name, expected_mean in means.items()):
            raise ValueError(f"Morphology aggregate mismatch: {key}")
        shape_audit[key] = {"rows": len(records), **means,
                            "row_indices_ordered_against_c0": True}
    for count in (26, 28):
        receipt = read(root / f"autodl_backup_readout_training_01/results/preflight_{count}.json")
        if (receipt["three_steps"]["steps"] != 3 or not all(receipt["three_steps"]["updated"].values())
                or receipt["random_label_validation_accuracy"] >= .4 or len(receipt["random_label_history"]) != 30):
            raise ValueError(f"Auxiliary readout preflight failed: {count}")
    expected = {"train": 22279, "validation": 2475, "test": 4157}
    if len(axis) != 36 or len(recheck) != 36:
        raise ValueError("Missing condition or role")
    for key, value in axis.items():
        if value["count"] != expected[key.split("/")[-1]] or recheck[key]["rows"] != value["count"]:
            raise ValueError(f"Population mismatch: {key}")
        if (recheck[key]["different_mirror_commands"] or recheck[key]["different_predictions"]
                or value["representative_readout_differences"]):
            raise ValueError(f"Recheck differences: {key}")
    for item in readout_audit.values():
        if any(run["differences"] for run in item["runs"]):
            raise ValueError("Auxiliary readout CPU prediction differs")
    if any(item["differences"] for item in pupil_audit.values()):
        raise ValueError("Pupil readout CPU prediction differs")
    statuses = {"C2_PREFLIGHT": "PASS", "C2_DMD_GRID_MAPPING": "PASS",
                "C2_BINARY_ENCODING": "PASS", "C2_PIXEL_FILL_FACTOR": "PASS",
                "C2_COHERENT_DMD_MODEL": "BLOCKED", "C2_IDEAL_RELAY": "PASS",
                "C2_FINITE_PUPIL": "PARTIAL_PASS", "C2_DEFOCUS_ALIGNMENT": "PARTIAL_PASS",
                "C2_ENERGY_ACCOUNTING": "PARTIAL_PASS", "C2_IMAGE_QUALITY": "PARTIAL_PASS",
                "C2_MTF": "PARTIAL_PASS", "C2_FROZEN_LINEAR_READOUT": "PASS",
                "C2_RETRAINED_READOUT_DIAGNOSTIC": "PASS", "C2_SENSITIVITY": "PARTIAL_PASS",
                "C2_TESTS": f"PASS_CROSSSTAGE_{tests-skipped}_AND_C2_FOCUSED_{focused_tests-focused_skipped}",
                "C2_REPORT": "PARTIAL_PASS",
                "C2_HARDWARE_CALIBRATION": "BLOCKED", "HW207_CURRENT_MAPPING": "BLOCKED"}
    summary = {"status": "C2_PARTIAL_PASS", "statuses": statuses, "axis_population": expected,
               "axis_condition_count": 12, "recomputed_condition_rows": sum(v["rows"] for v in recheck.values()),
               "assumptions": ["28 mirrors: M=200/(28*7.56); 26 mirrors: M=1",
                               "532 nm, 200 um target, 100x100 pixels", "0.92 square active area",
                               "uniform illumination and zero ON phase", "cutoffs and alignment illustrative"],
               "axis_conditions": axis, "auxiliary_selection": selection, "auxiliary_test": readout,
               "pupil_subset": pupil, "pupil_refinement": refine, "pupil_energy": ledger,
               "pupil_population_audit": full_pupil, "pupil_repeat": repeat,
               "pupil_simulated_interruption_resume": simulated_resume,
               "threshold_validation": threshold, "axis_sensitivity_validation": sensitivity,
               "rotation_validation": rotation, "grating_response": grating, "shape_subset": shapes,
               "shape_population": {"identity": shape_population["identity"],
                                    "definition": shape_population["definition"],
                                    "condition_count": len(shape_audit),
                                    "condition_sample_records": sum(v["rows"] for v in shape_audit.values()),
                                    "role_population": {"validation": 2475, "test": 4157},
                                    "conditions": shape_audit,
                                    "artifact": "autodl_backup_shape_population_03/shape_population.json",
                                    "c0_indices_artifact": "autodl_backup_c0_inputs_01/indices.json",
                                    "c0_data_manifest_artifact": "autodl_backup_c0_inputs_01/data.json"},
               "frozen_test_margins": margins,
               "test_evidence": {"crossstage_final_03": {"tests": tests, "passed": tests - skipped,
                                                          "skipped": skipped,
                                                          "artifact": "test_crossstage_final_03/results.xml"},
                                 "c2_focused": {"tests": focused_tests, "passed": focused_tests - focused_skipped,
                                                "skipped": focused_skipped,
                                                "artifact": "test_c2_directed_02/results.xml"}},
               "limitations": ["finite-pupil validation uses 2475 samples and fixed baseline test uses 4157; refinement uses 4",
                               "coherent grating contrast proxy is not general incoherent MTF",
                               "official test was previously used in C0/C1",
                               "morphology is a full-population fixed-threshold numerical proxy, not a physical image-quality measurement",
                               "real device identity, collection, relay magnification and NA remain uncalibrated",
                               "no complete tilted-mirror coherent DMD model"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
