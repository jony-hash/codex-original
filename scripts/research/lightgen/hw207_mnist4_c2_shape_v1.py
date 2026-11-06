"""Fixed-threshold stroke and connectivity proxies on saved validation images."""
import argparse
import json
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c0_v1 import features, map_image
from hw207_mnist4_c1_v1 import population
from hw207_mnist4_c2_v1 import DMD_PITCH_UM, axis_aligned_pixel_integrals, encode


def components(image):
    mask = image.tolist()
    height, width = image.shape
    seen = set()
    total = 0
    for y in range(height):
        for x in range(width):
            if not mask[y][x] or (y, x) in seen:
                continue
            total += 1
            seen.add((y, x))
            pending = deque([(y, x)])
            while pending:
                yy, xx = pending.popleft()
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    point = (yy+dy, xx+dx)
                    if (0 <= point[0] < height and 0 <= point[1] < width
                            and mask[point[0]][point[1]] and point not in seen):
                        seen.add(point)
                        pending.append(point)
    return total


def metrics(reference, candidate, thin_reference=None):
    """Threshold candidate at half its own maximum; report four-neighbor proxies."""
    reference = reference.bool()
    maximum = candidate.max()
    mask = candidate >= .5 * maximum if maximum > 0 else torch.zeros_like(reference)
    padded = F.pad(reference.to(torch.int16)[None, None], (1, 1, 1, 1))
    neighbors = padded[:, :, :-2, 1:-1] + padded[:, :, 2:, 1:-1] + padded[:, :, 1:-1, :-2] + padded[:, :, 1:-1, 2:]
    thin = reference & (neighbors[0, 0] <= 2) if thin_reference is None else thin_reference.bool()
    ref_n, out_n = int(reference.sum()), int(mask.sum())
    return {"foreground_ratio": out_n/ref_n if ref_n else None,
            "dice": 2 * int((reference & mask).sum()) / (ref_n + out_n) if ref_n + out_n else None,
            "reference_components": components(reference), "candidate_components": components(mask),
            "thin_pixel_retention": float((thin & mask).sum()) / int(thin.sum()) if thin.any() else None}


def main():
    p = argparse.ArgumentParser()
    for name in ("indices", "data", "train", "test", "input", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images, labels, rows = population(args.indices, args.data, args.train, args.test)["validation"]
    selection = torch.cat([(labels == k).nonzero().flatten()[:4] for k in range(4)]).sort().values
    images, labels, rows = images[selection], labels[selection], rows[selection]
    source = features(images, True)
    reference = map_image(source, True).bool()
    cases = {}
    for count, mag in ((26, 1.), (28, 200/(28*DMD_PITCH_UM))):
        for method in ("nearest", "area"):
            commands = encode(source, count, method, magnification=mag)
            cases[f"ideal_{count}_{method}"] = axis_aligned_pixel_integrals(commands, mag, .92)["mean_intensity"]
    for key in ("cutoff_0.05_defocus_0.0", "cutoff_0.1_defocus_0.0",
                "cutoff_0.2_defocus_0.0", "cutoff_0.1_defocus_50.0"):
        data = torch.load(args.input / f"{key}.pt", map_location="cpu", weights_only=True)
        if not torch.equal(data["rows"], rows):
            raise ValueError("Saved sample rows do not match")
        cases[key] = data["intensity"]
    report = {"identity": "FIXED_THRESHOLD_SHAPE_PROXY_VALIDATION_SUBSET",
              "definition": "candidate >= 0.5 * per-image maximum; thin = <=2 four-neighbors",
              "rows": rows.tolist(), "labels": labels.tolist(), "conditions": {}}
    for key, values in cases.items():
        result = [metrics(a, b) for a, b in zip(reference, values)]
        report["conditions"][key] = {"per_row": result,
            "mean_dice": sum(r["dice"] for r in result) / len(result),
            "mean_thin_pixel_retention": sum(r["thin_pixel_retention"] for r in result) / len(result),
            "mean_component_change": sum(r["candidate_components"]-r["reference_components"] for r in result)/len(result)}
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
