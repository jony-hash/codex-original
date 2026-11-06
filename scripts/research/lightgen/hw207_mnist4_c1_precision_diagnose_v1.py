"""Read-only localization of CPU/GPU C1 linear readout differences."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c1_v1 import population


def main():
    parser = argparse.ArgumentParser()
    for key in ("indices", "data", "train", "test", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    samples = population(args.indices, args.data, args.train, args.test)
    _, labels, rows = samples["test"]
    root = args.output
    selection = json.loads((root / "readout_selection.json").read_text())
    saved = json.loads((root / "readout_test.json").read_text())
    key = "gray/0.001/25/raw"
    chosen = selection[key]
    candidate = torch.load(root / "readouts/gray/0.001/25/raw" / f"decay_{chosen['decay']}_seed_11.pt",
                           map_location="cpu", weights_only=True)
    state = candidate["state"]
    records = []
    all_images = []
    for start in range(0, len(rows), 256):
        end = min(start + 256, len(rows))
        chunk = torch.load(root / "chunks/gray/0.001/test" / f"{start:06d}.pt", weights_only=True)
        image = chunk["intensity"]
        all_images.append(image)
        x = F.avg_pool2d(image[:, None], 4)[:, 0].flatten(1) / chosen["scale"]
        cpu = x @ state["weight"].T + state["bias"]
        gpu_image = image.cuda()
        gpu_x = F.avg_pool2d(gpu_image[:, None], 4)[:, 0].flatten(1) / chosen["scale"]
        gpu = (gpu_x @ state["weight"].cuda().T + state["bias"].cuda()).cpu()
        precise = x.double() @ state["weight"].double().T + state["bias"].double()
        baseline = torch.tensor(saved[key]["runs"][0]["metrics"]["prediction"][start:end])
        mismatches = torch.where(cpu.argmax(1) != baseline)[0]
        for i in mismatches.tolist():
            records.append({"test_position": start+i, "original_index": rows[start+i].item(),
                            "label": labels[start+i].item(), "saved_prediction": baseline[i].item(),
                            "cpu_prediction": cpu[i].argmax().item(), "gpu_prediction": gpu[i].argmax().item(),
                            "fp64_prediction": precise[i].argmax().item(),
                            "cpu_logits": cpu[i].tolist(), "gpu_logits": gpu[i].tolist(),
                            "fp64_logits": precise[i].tolist(),
                            "input_pool_max_error": (x[i] - gpu_x[i].cpu()).abs().max().item()})
    full_image = torch.cat(all_images).cuda()
    full_x = F.avg_pool2d(full_image[:, None], 4)[:, 0].flatten(1) / chosen["scale"]
    model = torch.nn.Linear(625, 4, device="cuda")
    model.load_state_dict(state)
    with torch.no_grad():
        full_logits = model(full_x).cpu()
    original = torch.tensor(saved[key]["runs"][0]["metrics"]["prediction"])
    for record in records:
        position = record["test_position"]
        record["gpu_full_logits"] = full_logits[position].tolist()
        record["gpu_full_prediction"] = full_logits[position].argmax().item()
    print(json.dumps({"mismatches": records,
                      "gpu_full_vs_saved_count": (full_logits.argmax(1) != original).sum().item()}, indent=2))


if __name__ == "__main__":
    main()
