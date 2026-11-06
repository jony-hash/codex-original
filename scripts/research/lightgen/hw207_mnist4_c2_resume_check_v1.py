"""Create an isolated interrupted-run copy and verify its recovered chunk."""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--copy", type=Path, required=True)
    p.add_argument("--phase", choices=("setup", "verify"), required=True)
    args = p.parse_args()
    relative = Path("test/cutoff_0.1_defocus_0.0/001024.pt")
    original = args.source / relative
    clone = args.copy / relative
    if args.phase == "setup":
        if not original.is_file() or args.copy.exists():
            raise ValueError("Expected original chunk and new isolated destination")
        def omit_one(directory, names):
            return [relative.name] if Path(directory).resolve() == original.parent.resolve() else []
        shutil.copytree(args.source, args.copy, copy_function=os.link, ignore=omit_one)
        if clone.exists() or not original.is_file():
            raise ValueError("Interrupted-run setup failed")
        print("INTERRUPTED_COPY_READY", flush=True)
        return
    if not clone.is_file():
        raise ValueError("Missing recovered chunk")
    before = torch.load(original, map_location="cpu", weights_only=True)
    after = torch.load(clone, map_location="cpu", weights_only=True)
    for name in ("role", "cutoff", "defocus"):
        if before[name] != after[name]:
            raise ValueError(f"Recovered condition differs: {name}")
    for name in ("rows", "labels", "intensity", "pattern", "nrmse", "correlation", "target_sum"):
        if not torch.equal(before[name], after[name]):
            raise ValueError(f"Recovered data differ: {name}")
    for mode in ("raw", "normalized"):
        if not torch.equal(before["scores"][mode]["zero"], after["scores"][mode]["zero"]):
            raise ValueError("Recovered zero-energy markers differ")
        for a, b in zip(before["scores"][mode]["runs"], after["scores"][mode]["runs"]):
            if a["seed"] != b["seed"] or any(not torch.equal(a[k], b[k]) for k in ("prediction", "margin")):
                raise ValueError("Recovered readout differs")
    receipt = {"status": "PASS", "scope": "one missing test chunk in an isolated hardlink copy",
               "original_chunk": str(original), "recovered_chunk": str(clone),
               "rows": len(before["rows"]), "intensity_pattern_metrics_and_predictions": "IDENTICAL"}
    with (args.copy.parent / "resume_check.json").open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print("INTERRUPTED_RESUME_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
