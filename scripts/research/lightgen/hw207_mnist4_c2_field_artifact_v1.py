"""Save a representative complex field with explicit target coordinates."""
import argparse
import json
from pathlib import Path

import torch

from hw207_mnist4_c0_v1 import features
from hw207_mnist4_c1_v1 import population
from hw207_mnist4_c2_psf_v1 import pupil_convolve
from hw207_mnist4_c2_pupil_v1 import padded_axis_field
from hw207_mnist4_c2_v1 import DMD_PITCH_UM, encode


def main():
    p = argparse.ArgumentParser()
    for name in ("indices", "data", "train", "test", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA required and output must be new")
    args.output.mkdir(parents=True)
    images, labels, rows = population(args.indices, args.data, args.train, args.test)["validation"]
    image, label, row = images[:1], labels[:1], rows[:1]
    mag = 200/(28*DMD_PITCH_UM)
    pattern = encode(features(image, True).cuda(), 28, "nearest", magnification=mag)
    field = padded_axis_field(pattern, mag, .92, .25, 400).to(torch.complex64)
    output = pupil_convolve(field, .25, .1, 0.)
    meta = {"identity": "ASSUMED_SCALAR_COHERENT_FIELD_NOT_MEASURED", "role": "validation",
            "original_index": int(row[0]), "label": int(label[0]), "shape": [1600, 1600],
            "pitch_um": .25, "window_um": 400, "pixel_center_origin_um": [-199.875, -199.875],
            "axis_order": ["y", "x"], "wavelength_um": .532, "magnification": mag,
            "fill_factor": .92, "cutoff_cycles_per_um": .1, "defocus_um": 0.,
            "complex_field_units": "relative amplitude normalized by magnification"}
    torch.save({"input_field": field[0].cpu(), "output_field": output[0].cpu(),
                "pattern": pattern[0].cpu(), "metadata": meta}, args.output / "representative_field.pt")
    with (args.output / "field_metadata.json").open("x", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
