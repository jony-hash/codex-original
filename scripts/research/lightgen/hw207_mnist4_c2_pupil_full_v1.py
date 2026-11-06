"""Chunked pupil validation population and pre-fixed baseline test diagnostic."""
import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c0_v1 import features, map_image
from hw207_mnist4_c1_v1 import population, save_chunk
from hw207_mnist4_c2_psf_v1 import pupil_convolve, pupil_transfer
from hw207_mnist4_c2_pupil_v1 import padded_axis_field
from hw207_mnist4_c2_v1 import DMD_PITCH_UM, bind_identity_binary_heads, encode, frozen_scores, intensity_metrics


def main():
    p = argparse.ArgumentParser()
    for name in ("indices", "data", "train", "test", "c1-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    samples = population(args.indices, args.data, args.train, args.test)
    heads = bind_identity_binary_heads(args.c1_root, torch.device("cuda"))
    config = {"identity": "ASSUMED_SPATIAL_PSF_PUPIL_FULL_VALIDATION",
              "input": "28/nearest/0.92", "pitch_um": .25, "source_window_um": 400,
              "roles": {"validation": [[.05, 0.], [.1, 0.], [.2, 0.], [.1, 50.]],
                        "test": [[.1, 0.]]},
              "test_condition": "fixed before validation population calculation; no test-driven selection",
              "heads": {mode: [{"seed": h["seed"], "sha256": h["sha256"]} for h in info["models"]]
                        for mode, info in heads.items()}}
    config_path = args.output / "configuration.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Existing configuration mismatch")
    else:
        with config_path.open("x", encoding="utf-8") as f:
            json.dump(config, f, indent=2, allow_nan=False)
    mag = 200 / (28 * DMD_PITCH_UM)
    # Test condition is fixed in configuration before any population is processed.
    for role in ("validation", "test"):
        images, labels, rows = samples[role]
        for cutoff, defocus in config["roles"][role]:
            key = f"cutoff_{cutoff}_defocus_{defocus}"
            directory = args.output / role / key
            directory.mkdir(parents=True, exist_ok=True)
            transfer = pupil_transfer(1600, .25, cutoff, defocus, device="cuda")
            for start in range(0, len(rows), 32):
                end = min(start + 32, len(rows))
                path = directory / f"{start:06d}.pt"
                if path.exists():
                    existing = torch.load(path, map_location="cpu", weights_only=True)
                    if ((existing["role"], existing["cutoff"], existing["defocus"]) != (role, cutoff, defocus)
                            or not torch.equal(existing["rows"], rows[start:end])
                            or not torch.equal(existing["labels"], labels[start:end])
                            or existing["intensity"].shape != (end-start, 100, 100)):
                        raise ValueError(f"Existing chunk mismatch: {path}")
                    continue
                image = features(images[start:end], True).cuda()
                command = encode(image, 28, "nearest", magnification=mag)
                pieces = []
                for part in command.split(2):
                    field = padded_axis_field(part, mag, .92, .25, 400).to(torch.complex64)
                    output = pupil_convolve(field, .25, cutoff, defocus, transfer=transfer)
                    pieces.append(F.avg_pool2d(output[:, None, 400:1200, 400:1200].abs().square(), 8)[:, 0].cpu())
                intensity = torch.cat(pieces)
                reference = map_image(image, True).cpu()
                m = intensity_metrics(reference, intensity)
                scores = {}
                for mode in ("raw", "normalized"):
                    results, zeros = frozen_scores(intensity, heads[mode], mode == "normalized")
                    scores[mode] = {"zero": zeros.cpu(), "runs": [
                        {"seed": r["seed"], "prediction": r["prediction"].cpu(), "margin": r["margin"].cpu()}
                        for r in results]}
                if shutil.disk_usage(args.output).free < 1024**3:
                    raise OSError("Stopped before dropping below 1 GiB free")
                save_chunk(path, {"role": role, "cutoff": cutoff, "defocus": defocus,
                                  "rows": rows[start:end].clone(), "labels": labels[start:end].clone(),
                                  "intensity": intensity, "pattern": command.cpu().to(torch.uint8),
                                  "nrmse": m["nrmse"], "correlation": m["normalized_correlation"],
                                  "target_sum": m["candidate_sum"], "scores": scores})
                print(f"PUPIL_FULL {role} {key} {end}/{len(rows)}", flush=True)
            del transfer
    print("PUPIL_FULL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
