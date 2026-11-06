"""Compare independently executed subset and full-pipeline pupil intensities."""
import argparse
import io
import json
import tarfile
from pathlib import Path

import torch

from hw207_mnist4_c1_v1 import population


def main():
    p = argparse.ArgumentParser()
    for name in ("indices", "data", "train", "test", "input", "archive", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    _, _, population_rows = population(args.indices, args.data, args.train, args.test)["validation"]
    positions = {int(row): i for i, row in enumerate(population_rows)}
    report = {"identity": "REPEATED_PUPIL_SUBSET_VS_FULL", "conditions": {}}
    with tarfile.open(args.archive, "r") as archive:
        for key in ("cutoff_0.05_defocus_0.0", "cutoff_0.1_defocus_0.0",
                    "cutoff_0.2_defocus_0.0", "cutoff_0.1_defocus_50.0"):
            subset = torch.load(args.input / f"{key}.pt", map_location="cpu", weights_only=True)
            chunks = {}
            maxima = []
            for i, row in enumerate(subset["rows"].tolist()):
                pos = positions[row]
                start = (pos // 32) * 32
                if start not in chunks:
                    member = archive.extractfile(f"validation/{key}/{start:06d}.pt")
                    if member is None:
                        raise ValueError("Missing saved full-population chunk")
                    chunks[start] = torch.load(io.BytesIO(member.read()), map_location="cpu", weights_only=True)
                chunk = chunks[start]
                offset = pos-start
                if int(chunk["rows"][offset]) != row:
                    raise ValueError("Population order mismatch")
                maxima.append(float((chunk["intensity"][offset]-subset["intensity"][i]).abs().max()))
            report["conditions"][key] = {"rows": len(maxima), "max_abs_difference": max(maxima)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
