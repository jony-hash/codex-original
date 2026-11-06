"""Fixed mapping comparison with thin pixels defined on original 28x28 images."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c0_v1 import features, map_image
from hw207_mnist4_c1_v1 import population
from hw207_mnist4_c2_shape_v1 import metrics
from hw207_mnist4_c2_v1 import axis_aligned_pixel_integrals


def original_thin(source):
    binary = source.bool()
    padded = F.pad(binary.to(torch.int16)[:, None], (1, 1, 1, 1))
    neighbors = (padded[:, :, :-2, 1:-1] + padded[:, :, 2:, 1:-1] +
                 padded[:, :, 1:-1, :-2] + padded[:, :, 1:-1, 2:])[:, 0]
    thin = binary & (neighbors <= 2)
    return map_image(thin.float(), True).bool()


def main():
    p = argparse.ArgumentParser()
    for name in ("indices", "data", "train", "test", "input", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    samples = population(args.indices, args.data, args.train, args.test)
    report = {"identity": "FIXED_THRESHOLD_ORIGINAL_GRID_THINNESS_VALIDATION_TEST",
              "definition": "thin on 28x28: foreground with <=2 four-neighbors, then nearest to 100x100; candidate >= 0.5 max",
              "conditions": {}}
    for count, mag in ((26, 1.), (28, 200/(28*7.56))):
        for method in ("nearest", "area"):
            for role in ("validation", "test"):
                images, labels, rows = samples[role]
                records = []
                for start in range(0, len(rows), 256):
                    end = min(start+256, len(rows))
                    path = args.input / "axis" / str(count) / method / "0.92" / role / f"{start:06d}.pt"
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if not torch.equal(saved["rows"], rows[start:end]) or not torch.equal(saved["labels"], labels[start:end]):
                        raise ValueError("Sample identity mismatch")
                    source = features(images[start:end], True)
                    ref = map_image(source, True).bool()
                    thin = original_thin(source)
                    output = axis_aligned_pixel_integrals(saved["pattern"], mag, .92)["mean_intensity"]
                    for i in range(end-start):
                        result = metrics(ref[i], output[i], thin[i])
                        if result["thin_pixel_retention"] is None or result["dice"] is None:
                            raise ValueError("Undefined shape statistic")
                        result["row"] = int(rows[start+i])
                        records.append(result)
                key = f"{count}/{method}/0.92/{role}"
                report["conditions"][key] = {"rows": len(records),
                    "mean_dice": sum(v["dice"] for v in records)/len(records),
                    "mean_thin_pixel_retention": sum(v["thin_pixel_retention"] for v in records)/len(records),
                    "mean_component_change": sum(v["candidate_components"]-v["reference_components"] for v in records)/len(records),
                    "per_row": records}
                print(f"SHAPE_POPULATION {key} rows={len(records)}", flush=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
