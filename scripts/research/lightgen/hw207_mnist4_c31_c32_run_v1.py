"""Run the C3.1 ideal-screen CMOS diagnostic on the frozen C0 validation split."""
import argparse
import json
import struct
import time
from pathlib import Path

import torch
import numpy as np

from scripts.research.lightgen.hw207_mnist4_c31_c32_cmos_v1 import prepare_configuration, forward_hw207_mnist4


def load_inputs(root, train_pt=None, indices_path=None):
    if train_pt is None:
        images = (root / 'data/MNIST/raw/train-images-idx3-ubyte').read_bytes()
        labels = (root / 'data/MNIST/raw/train-labels-idx1-ubyte').read_bytes()
        if struct.unpack('>IIII', images[:16]) != (2051, 60000, 28, 28) or struct.unpack('>II', labels[:8]) != (2049, 60000):
            raise ValueError('Unexpected MNIST training IDX headers')
        x = torch.from_numpy(np.frombuffer(images, dtype=np.uint8, offset=16).copy().reshape(60000, 28, 28))
        y = torch.from_numpy(np.frombuffer(labels, dtype=np.uint8, offset=8).copy()).long()
    else:
        x, y = torch.load(train_pt, map_location='cpu', weights_only=True)
        if x.shape != (60000, 28, 28) or x.dtype != torch.uint8 or y.shape != (60000,) or y.dtype != torch.int64:
            raise ValueError('Unexpected MNIST training tensor format')
    p = indices_path or root / 'outputs/20261103_hw207_mnist4_c2_v1/runs/autodl_backup_c0_inputs_01/indices.json'
    indices = json.loads(p.read_text(encoding='utf-8'))
    train = torch.tensor(indices['train']['original_indices'])
    val = torch.tensor(indices['validation']['original_indices'])
    if len(train) != 22279 or len(val) != 2475 or len(torch.unique(torch.cat((train, val)))) != 24754:
        raise ValueError('C0 training/validation roles or indices invalid')
    if torch.bincount(y[val], minlength=4).tolist() != [592, 674, 596, 613]:
        raise ValueError('C0 validation labels do not match')
    return x, y, val


def metrics(labels, predictions, scores, total):
    cm = torch.zeros((4, 5), dtype=torch.int64)
    for actual, pred in zip(labels.tolist(), predictions.tolist(), strict=True):
        cm[actual, pred if pred >= 0 else 4] += 1
    tp = cm[:, :4].diag().double()
    precision = tp / cm[:, :4].sum(0).clamp_min(1)
    recall = tp / cm.sum(1).clamp_min(1)
    f1 = 2 * tp / (cm.sum(1) + cm[:, :4].sum(0)).clamp_min(1)
    correct = scores[torch.arange(len(labels)), labels]
    wrong = scores.clone(); wrong[torch.arange(len(labels)), labels] = -1
    return {'accuracy': float((predictions == labels).double().mean()), 'macro_f1': float(f1.mean()),
            'precision': precision.tolist(), 'recall': recall.tolist(), 'confusion_4x4': cm[:, :4].tolist(),
            'abstentions_by_class': cm[:, 4].tolist(), 'coverage': float((predictions >= 0).double().mean()),
            'correct_roi_fraction': float((correct / total.clamp_min(1e-30)).mean()),
            'max_wrong_roi_fraction': float((wrong.max(1).values / total.clamp_min(1e-30)).mean()),
            'correct_minus_wrong_fraction': float(((correct - wrong.max(1).values) / total.clamp_min(1e-30)).mean()),
            'outside_fraction': float(((total - scores.sum(1)).clamp_min(0) / total.clamp_min(1e-30)).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['pilot', 'validation'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--defocus-um', type=float, default=0.0)
    parser.add_argument('--train-pt', type=Path)
    parser.add_argument('--indices', type=Path)
    parser.add_argument('--candidate-config', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    x, y, val = load_inputs(root, args.train_pt, args.indices)
    if args.phase == 'pilot':
        rng = torch.Generator().manual_seed(207031)
        chosen = [torch.where(y[val] == k)[0][torch.randperm(int((y[val] == k).sum()), generator=rng)[:64]] for k in range(4)]
        rows = val[torch.sort(torch.cat(chosen)).values]
    else:
        if not (args.output / 'pilot' / 'summary.json').is_file():
            raise FileNotFoundError('Pilot summary is required before validation')
        rows = val
    out = args.output / args.phase
    out.mkdir(parents=True, exist_ok=False)
    candidate_path = args.candidate_config or root / 'configs/research/lightgen/hw207_mnist4_c21_c3_spatial_v1.json'
    candidates = json.loads(candidate_path.read_text(encoding='utf-8'))['relay_candidates']
    summary = {'phase': args.phase, 'rows': len(rows), 'official_test_evaluated': False, 'defocus_um': args.defocus_um,
               'device': args.device, 'conditions': {}, 'model': 'IDEAL_SINGLE_PHASE_ONLY_SURFACE_ZERO_PHASE'}
    for candidate in candidates:
        started = time.perf_counter()
        context = prepare_configuration(candidate, args.device, args.defocus_um)
        scores, predictions, totals, inputs, outputs = [], [], [], [], []
        example_positions = torch.cat([torch.where(y[rows] == k)[0][:5] for k in range(4)])
        selected = set(example_positions.tolist())
        examples, example_labels, example_indices, example_inputs, example_outputs = [], [], [], [], []
        class_sums = torch.zeros((4, 768, 768), dtype=torch.float64)
        class_counts = torch.zeros(4, dtype=torch.int64)
        for start in range(0, len(rows), args.batch):
            ids = rows[start:start + args.batch]
            result = forward_hw207_mnist4(x[ids], context)
            scores.append(result['roi_scores'].cpu()); predictions.append(result['prediction'].cpu())
            totals.append(result['detected_energy'].cpu())
            inputs.append(result['energy']['lightgen_input_energy_um2'].cpu())
            outputs.append(result['energy']['relay_output_energy_um2'].cpu())
            pixels = result['cmos_pixel_integral'].cpu()
            for label in range(4):
                mask = y[ids] == label
                class_sums[label] += pixels[mask].sum(0)
                class_counts[label] += mask.sum()
            for position in range(len(ids)):
                if start + position in selected:
                    examples.append(pixels[position].float())
                    example_labels.append(y[ids[position]])
                    example_indices.append(ids[position])
                    example_inputs.append(result['lightgen_input_field'][position].cpu())
                    example_outputs.append(result['lightgen_output_field'][position].cpu())
        scores, predictions, totals = map(torch.cat, (scores, predictions, totals))
        result = metrics(y[rows], predictions, scores, totals)
        result.update({'rows': len(rows), 'roi_bounds_tile': context.rois,
                       'roi_bounds_sensor': [[v + 640 for v in rect] for rect in context.rois],
                       'output_magnification': context.magnification,
                       'mean_input_energy': float(torch.cat(inputs).mean()),
                       'mean_relay_output_energy': float(torch.cat(outputs).mean()),
                       'mean_detected_energy': float(totals.mean()), 'seconds': time.perf_counter() - started})
        torch.save({'indices': rows, 'labels': y[rows], 'scores': scores, 'predictions': predictions,
                    'detected_energy': totals, 'example_indices': torch.stack(example_indices),
                    'example_labels': torch.stack(example_labels), 'example_cmos': torch.stack(examples),
                    'example_lightgen_input': torch.stack(example_inputs),
                    'example_lightgen_output': torch.stack(example_outputs),
                    'class_mean_cmos': (class_sums / class_counts[:, None, None]).float()},
                   out / (candidate['id'] + '.pt'))
        summary['conditions'][candidate['id']] = result
        print(candidate['id'], json.dumps(result, ensure_ascii=False), flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
