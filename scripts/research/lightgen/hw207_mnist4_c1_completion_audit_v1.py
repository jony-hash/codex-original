"""Verify the saved C1 closeout evidence against the authorized population."""
import json
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    root = Path("outputs/20261103_hw207_mnist4_c1_v1/runs")
    data = root / "autodl_backup_data_01"
    models = root / "autodl_backup_readouts_01"
    controls = root / "autodl_backup_controls_01"
    convergence = read(root / "autodl_backup_preflight_01/convergence.json")
    preflight = read(data / "propagation_preflight.json")
    local = read(data / "local_independent_v2.json")
    cloud = read(root / "autodl_backup_verify_01/cloud_independent_v2.json")
    selection = read(models / "readout_selection.json")
    results = read(models / "readout_test.json")
    if read(data / "transfer_receipt.json")["status"] != "COMPLETE":
        raise ValueError("Incomplete data backup")
    if read(models / "transfer_receipt.json")["status"] != "COMPLETE":
        raise ValueError("Incomplete model backup")
    if read(controls / "transfer_receipt.json")["status"] != "COMPLETE":
        raise ValueError("Incomplete job control backup")
    for name in ("convergence", "propagation", "readout", "roi_audit", "verify_v2"):
        if read(controls / (name + "_status.json"))["state"] != "COMMAND_EXITED_ZERO":
            raise ValueError(f"Nonzero or missing terminal status: {name}")
    if read(controls / "verify_first_status.json")["state"] != "COMMAND_EXITED_NONZERO":
        raise ValueError("Initial failed verification evidence missing")
    if preflight["status"] != "READY_PROPAGATION" or not preflight["probes"]["passed"]:
        raise ValueError("Propagation preflight incomplete")
    if len(preflight["readout_checks"]) != 16:
        raise ValueError("Missing readout preflight")
    for size in (25, 100):
        if preflight["readout_checks"][f"binary_0.001_{size}_False"]["negative_accuracy"] >= 0.4:
            raise ValueError("Negative control rejected")
    for name in ("gray", "binary"):
        for distance in ("0.0005", "0.001", "0.002"):
            item = convergence["results"][name][distance]
            final = item["comparisons"][-1]
            if not final["passed"] or not item["precision"]["passed"]:
                raise ValueError("Convergence failed")
            for field in ("image_errors", "roi_errors"):
                if len(final[field]) != 64 or max(final[field]) > 1e-3:
                    raise ValueError("Invalid convergence population or error")
            for field in ("complex_errors", "energy_errors"):
                if len(item["precision"][field]) != 64 or max(item["precision"][field]) > 1e-5:
                    raise ValueError("Invalid precision population or error")
    if len(list((data / "chunks").rglob("*.pt"))) != 920:
        raise ValueError("Incomplete chunk population")
    if len(list((data / "roi_audit").rglob("*.pt"))) != 24:
        raise ValueError("Incomplete ROI audit")
    if len(selection) != 16 or len(results) != 16 or len(list((models / "readouts").rglob("*.pt"))) != 144:
        raise ValueError("Incomplete candidate models")
    if len(local["conditions"]) != 24 or len(local["readouts"]) != 16 or len(cloud["readouts"]) != 16:
        raise ValueError("Incomplete independent verification")
    for result in cloud["readouts"].values():
        if result["cuda_mismatches"] != [0, 0, 0]:
            raise ValueError("CUDA frozen predictions differ")
    for key, chosen in selection.items():
        runs = results[key]["runs"]
        if len(runs) != 3:
            raise ValueError(f"Missing final seeds: {key}")
        if [run["seed"] for run in runs] != [11, 23, 37]:
            raise ValueError(f"Wrong seed set: {key}")
        for run in runs:
            metrics = run["metrics"]
            if len(metrics["prediction"]) != 4157 or sum(metrics["support"]) != 4157:
                raise ValueError(f"Wrong test population: {key}")
        name, distance, size, mode = key.split("/")
        for decay in (0.0, 0.0001, 0.001):
            for seed in (11, 23, 37):
                path = models / "readouts" / name / distance / size / mode / f"decay_{decay}_seed_{seed}.pt"
                candidate = torch.load(path, map_location="cpu", weights_only=True)
                if candidate["seed"] != seed or candidate["decay"] != decay or len(candidate["history"]) != 30:
                    raise ValueError(f"Invalid training record: {path}")
                if [x["epoch"] for x in candidate["history"]] != list(range(1, 31)):
                    raise ValueError(f"Invalid training schedule: {path}")
        if chosen["decay"] not in (0.0, 0.0001, 0.001):
            raise ValueError(f"Invalid selection: {key}")
    report = Path("docs/research/lightgen/hw207_mnist4_c1_report.md")
    if not report.is_file() or "C1_NUMERICAL_DIAGNOSTIC_PASS" not in report.read_text(encoding="utf-8"):
        raise ValueError("Final report missing")
    audit = {"status": "COMPLETE", "conditions": 8, "population_per_condition": 28911,
             "data_roles": 24, "chunks": 920, "readout_configurations": 16, "candidate_models": 144,
             "preflight_sample_per_class": {"train": 128, "validation": 16},
             "cpu_disagreement_conditions": {k: v["differences"] for k, v in local["readouts"].items() if any(v["mismatches"])},
             "cuda_full_batch_mismatches": 0}
    target = data / "completion_audit.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
