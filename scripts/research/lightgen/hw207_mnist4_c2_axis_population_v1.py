"""C2 axis-aligned finite mirror diagnostic with frozen C1 readouts."""
import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c0_v1 import features, map_image
from hw207_mnist4_c1_v1 import population, save_chunk
from hw207_mnist4_c2_v1 import (
    axis_aligned_pixel_integrals, bind_identity_binary_heads, encode,
    frozen_scores, intensity_metrics,
)


def main():
    parser = argparse.ArgumentParser()
    for key in ("indices", "data", "train", "test", "c1_root", "output"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C2 production diagnostic requires CUDA")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    samples = population(args.indices, args.data, args.train, args.test)
    device = torch.device(args.device)
    heads = bind_identity_binary_heads(args.c1_root, device)
    validation = samples["validation"][0][:16]
    selected = encode(features(validation, True).to(device), 28, "nearest", magnification=200 / (28 * 7.56))
    exact = axis_aligned_pixel_integrals(selected, 200 / (28 * 7.56))
    if not torch.isfinite(exact["mean_intensity"]).all():
        raise ValueError("Nonfinite target intensity")
    full = torch.ones((1, 28, 28), device=device)
    target = axis_aligned_pixel_integrals(full, 200 / (28 * 7.56))["mean_intensity"]
    expected = 28 ** 2 * 7.56 ** 2
    relative_error = abs(target.sum().item() * 4 - expected) / expected
    if relative_error > 1e-10:
        raise ValueError("Analytic area/energy check failed")
    zero = axis_aligned_pixel_integrals(torch.zeros_like(full), 1.0)["mean_intensity"]
    if zero.count_nonzero():
        raise ValueError("OFF negative control failed")
    test_intensity = map_image(features(samples["test"][0][:256], True), True).to(device)
    prior = json.loads((args.c1_root / "readout_test.json").read_text(encoding="utf-8"))
    for mode in ("raw", "normalized"):
        observed, _ = frozen_scores(test_intensity, heads[mode], mode == "normalized")
        for index, result in enumerate(observed):
            expected_predictions = prior[f"binary/0/100/{mode}"]["runs"][index]["metrics"]["prediction"][:256]
            if result["prediction"].cpu().tolist() != expected_predictions:
                raise ValueError("Frozen identity binding mismatch")
    receipt = {"status": "READY_AXIS_ALIGNED_ONLY", "area_relative_error": relative_error,
               "zero_control": "PASS", "identity_readout_first_256": "PASS",
               "input_paths": {key: str(getattr(args, key).resolve())
                               for key in ("indices", "data", "train", "test", "c1_root")},
               "conditions": ["26 x 1.0", "28 x 200/(28*7.56)"],
               "assumptions": ["uniform illumination", "square mirror geometry",
                               "positive magnification orientation assumed",
                               "fill factor 0.92 is an illustrative diagnostic"],
               "hardware_mapping": "BLOCKED", "coherent_dmd": "BLOCKED",
               "heads": {mode: {"scale": value["scale"], "decay": value["decay"],
                                  "models": [{k: item[k] for k in ("seed", "sha256", "path")}
                                             for item in value["models"]]}
                         for mode, value in heads.items()}}
    args.output.mkdir(parents=True, exist_ok=args.resume)
    receipt_path = args.output / "axis_preflight.json"
    if receipt_path.exists():
        if json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
            raise ValueError("Existing preflight does not match inputs and frozen heads")
    else:
        with receipt_path.open("x", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, allow_nan=False)
    for count, magnification in ((26, 1.0), (28, 200 / (28 * 7.56))):
        for method in ("nearest", "area", "c0_nearest"):
            for fill in (1.0, 0.92):
                for role, (images, labels, rows) in samples.items():
                    directory = args.output / "axis" / str(count) / method / str(fill) / role
                    directory.mkdir(parents=True, exist_ok=True)
                    for start in range(0, len(rows), 256):
                        end = min(start + 256, len(rows))
                        chunk_path = directory / f"{start:06d}.pt"
                        if chunk_path.exists():
                            saved = torch.load(chunk_path, map_location="cpu", weights_only=True)
                            if ((saved["role"], saved["count"], saved["magnification"], saved["fill_factor"], saved["method"])
                                    != (role, count, magnification, fill, method)
                                    or not torch.equal(saved["rows"], rows[start:end])
                                    or not torch.equal(saved["labels"], labels[start:end])
                                    or saved["pattern"].shape != (end - start, count, count)):
                                raise ValueError(f"Existing chunk mismatch: {chunk_path}")
                            continue
                        intensity_parts, pattern_parts = [], []
                        for raw in images[start:end].split(32):
                            source = features(raw, True).to(device)
                            if method == "c0_nearest":
                                source = map_image(source, True)
                            command = encode(source, count, "nearest" if method == "c0_nearest" else method,
                                             magnification=magnification)
                            result = axis_aligned_pixel_integrals(command, magnification, fill)
                            pattern_parts.append(command.to(torch.uint8).cpu())
                            intensity_parts.append(result["mean_intensity"].float().cpu())
                        intensity = torch.cat(intensity_parts)
                        command = torch.cat(pattern_parts)
                        original = map_image(features(images[start:end], True), True)
                        metric = intensity_metrics(original, intensity)
                        outputs = {}
                        for mode in ("raw", "normalized"):
                            results, zeros = frozen_scores(intensity, heads[mode], mode == "normalized")
                            outputs[mode] = {"zero": zeros.cpu(),
                                "runs": [{"seed": result["seed"], "prediction": result["prediction"].cpu(),
                                          "margin": result["margin"].cpu()} for result in results]}
                        payload = {"role": role, "binary": True, "count": count,
                                   "magnification": magnification, "fill_factor": fill, "method": method,
                                   "rows": rows[start:end].clone(), "labels": labels[start:end].clone(),
                                   "pattern": command, "representative_intensity": intensity[:1].clone(),
                                   "intensity_storage": "first row only; full image analytically reproducible from saved pattern and geometry",
                                   "nrmse": metric["nrmse"], "normalized_correlation": metric["normalized_correlation"],
                                   "reference_sum": metric["reference_sum"], "candidate_sum": metric["candidate_sum"],
                                   "frozen": outputs}
                        if shutil.disk_usage(args.output).free < 1024 ** 3:
                            raise OSError("C2 output stopped with less than 1 GiB remaining")
                        save_chunk(chunk_path, payload)
                    print(f"COMPLETE {count} {method} {fill} {role} rows={len(rows)}", flush=True)
    print("AXIS_ALIGNED_POPULATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
