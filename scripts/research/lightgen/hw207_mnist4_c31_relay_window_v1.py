"""Audit output-relay field windows using saved C3.1 validation complex fields."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.relays import CoherentImagingRelay
from scripts.research.lightgen.hw207_mnist4_c31_c32_cmos_v1 import prepare_configuration


@torch.no_grad()
def relay_window(fields, candidate, context, window, device, include_field=False):
    m = context.magnification
    f1, f2 = (v * 1e-3 for v in candidate['output_focal_lengths_mm'])
    segments = (f1, f1 + f2, f2)
    source = PlaneSpec('window_source', (window, window), (2e-6, 2e-6), 532e-9, 0, 'window')
    target = source.with_plane('window_image', z_m=sum(segments), pixel_pitch_m=(abs(m)*2e-6,)*2)
    relay = CoherentImagingRelay(source, target, m, name='window_audit', segments_m=segments,
                                  focal_lengths_m=(f1, f2), clear_aperture_m=6.384e-3)
    pad = (window - 100) // 2
    field = F.pad(fields.to(device)[:, None], (pad, pad, pad, pad))
    output = relay(ComplexField(field, source, OperationProvenance('saved_complex_input', None, None))).tensor[:, 0]
    start = (window - 256) // 2
    crop = output[:, start:start+256, start:start+256]
    intensity = crop.abs().square()
    pixels = torch.sparse.mm(context.camera_matrix, intensity.flatten(1).double().T).T.reshape(-1, 768, 768)
    scores = torch.stack([pixels[:, r0:r1, c0:c1].sum((1, 2)) for r0,r1,c0,c1 in context.rois], 1)
    scale = (2 * abs(m)) ** 2
    full = output.abs().square().double().sum((1, 2)) * scale
    cropped = intensity.double().sum((1, 2)) * scale
    result = (scores.cpu(), full.cpu(), cropped.cpu())
    return result + (output.cpu(),) if include_field else result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    candidates=json.loads(Path('configs/research/lightgen/hw207_mnist4_c21_c3_spatial_v1.json').read_text(encoding='utf-8'))['relay_candidates']
    results={}
    for candidate in candidates:
        name=candidate['id']; saved=torch.load(args.data/(name+'.pt'),map_location='cpu',weights_only=True)
        context=prepare_configuration(candidate,args.device)
        fields=saved['example_lightgen_output']; labels=saved['example_labels']
        measured={}
        for window in (256,512):
            parts=[relay_window(x,candidate,context,window,args.device) for x in fields.split(2)]
            scores,energy,crop=(torch.cat([p[i] for p in parts]) for i in range(3))
            measured[window]={'scores':scores,'energy':energy,'crop':crop}
        a,b=measured[256],measured[512]
        denominator=fields.abs().square().double().sum((1,2))*4
        error=(a['scores']-b['scores']).abs().sum(1)/denominator.clamp_min(1e-30)
        previous=torch.stack([saved['scores'][torch.where(saved['indices']==i)[0].item()] for i in saved['example_indices']])
        prior_error=(previous-b['scores']).abs().sum(1)/denominator.clamp_min(1e-30)
        fp32=relay_window(fields[:1],candidate,context,512,args.device,True)
        fp64=relay_window(fields[:1].to(torch.complex128),candidate,context,512,args.device,True)
        field_relative_error=torch.linalg.vector_norm(fp32[3].to(torch.complex128)-fp64[3])/torch.linalg.vector_norm(fp64[3]).clamp_min(1e-30)
        results[name]={'rows':len(fields),'labels':labels.tolist(),'original_indices':saved['example_indices'].tolist(),
          'max_roi_l1_over_input_energy':float(error.max()), 'mean_roi_l1_over_input_energy':float(error.mean()),
          'max_prior100_to_window512_roi_l1_over_input_energy':float(prior_error.max()),
          'one_sample_fp32_fp64_complex_field_relative_l2':float(field_relative_error),
          'window256_accuracy':float((a['scores'].argmax(1)==labels).double().mean()),
          'window512_accuracy':float((b['scores'].argmax(1)==labels).double().mean()),
          'prior100_accuracy':float((previous.argmax(1)==labels).double().mean()),
          'window512_mean_outside_roi_fraction':float(((b['energy']-b['scores'].sum(1)).clamp_min(0)/b['energy'].clamp_min(1e-30)).mean()),
          'window512_mean_outside_256_crop_fraction':float(((b['energy']-b['crop']).clamp_min(0)/b['energy'].clamp_min(1e-30)).mean()),
          'decisions_changed_256_to_512':int((a['scores'].argmax(1)!=b['scores'].argmax(1)).sum()),
          'decisions_changed_prior_to_512':int((previous.argmax(1)!=b['scores'].argmax(1)).sum())}
        print(name,json.dumps(results[name]),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump({'status':'SAMPLED_WINDOW_DIAGNOSTIC','tolerance_for_review':1e-3,'conditions':results},stream,indent=2)


if __name__=='__main__': main()
