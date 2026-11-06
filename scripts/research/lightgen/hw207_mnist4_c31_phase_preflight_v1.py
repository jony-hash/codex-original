"""Three-step non-deployable ideal phase-screen gradient preflight."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.research.lightgen.hw207_mnist4_c31_c32_run_v1 import load_inputs
from scripts.research.lightgen.hw207_mnist4_c31_c32_cmos_v1 import prepare_configuration, forward_hw207_mnist4


def loss_for(images, labels, context):
    result = forward_hw207_mnist4.__wrapped__(images, context)
    scores = result['roi_scores']
    detected = result['detected_energy'].clamp_min(1e-20)
    input_energy = result['energy']['lightgen_input_energy_um2'].clamp_min(1e-20)
    logits = 20 * scores / detected[:, None]
    throughput = detected / input_energy
    loss = F.cross_entropy(logits.float(), labels) + .1 * F.relu(.8 - throughput).square().mean().float()
    return loss, result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda'); a=p.parse_args()
    root=Path.cwd(); images, labels, _=load_inputs(root)
    path=root/'outputs/20261103_hw207_mnist4_c2_v1/runs/autodl_backup_c0_inputs_01/indices.json'
    train=torch.tensor(json.loads(path.read_text(encoding='utf-8'))['train']['original_indices'])
    chosen=torch.cat([train[torch.where(labels[train]==k)[0][:2]] for k in range(4)])
    x=images[chosen]; y=labels[chosen].to(a.device)
    candidates=json.loads((root/'configs/research/lightgen/hw207_mnist4_c21_c3_spatial_v1.json').read_text(encoding='utf-8'))['relay_candidates']
    a.output.mkdir(parents=True,exist_ok=False); report={}
    for defocus in (0.,50.):
        context=prepare_configuration(candidates[2],a.device,defocus)
        context.screen.phase.requires_grad_(True)
        phase=context.screen.phase; optimizer=torch.optim.SGD([phase],lr=.1)
        loss,_=loss_for(x,y,context); loss.backward()
        grad=phase.grad.detach().clone(); index=int(grad.abs().flatten().argmax())
        analytic=float(grad.flatten()[index]); h=.05
        with torch.no_grad(): phase.flatten()[index]=h
        plus=float(loss_for(x,y,context)[0].detach())
        with torch.no_grad(): phase.flatten()[index]=-h
        minus=float(loss_for(x,y,context)[0].detach())
        with torch.no_grad(): phase.zero_()
        values=[]; norms=[]
        for _ in range(3):
            optimizer.zero_grad(); loss,result=loss_for(x,y,context); loss.backward()
            if not torch.isfinite(loss) or phase.grad is None or not torch.isfinite(phase.grad).all():
                raise ValueError('Nonfinite optical phase loss or gradient')
            values.append(float(loss.detach())); norms.append(float(phase.grad.norm())); optimizer.step()
        update=float(phase.detach().norm())
        name=f'defocus_{int(defocus)}um'
        finite_difference=(plus-minus)/(2*h)
        relative_error=abs(analytic-finite_difference)/max(abs(analytic),1e-12)
        passed=abs(analytic)>1e-6 and relative_error<.05 and update>1e-6
        report[name]={'loss':values,'gradient_norm':norms,'phase_update_l2':update,
          'strongest_gradient':analytic,'finite_difference':finite_difference,'gradient_relative_error':relative_error,
          'mean_detected_over_chip_input':float((result['detected_energy']/result['energy']['lightgen_input_energy_um2'].clamp_min(1e-20)).mean()),
          'status':'THREE_STEP_GRADIENT_CHECK_PASS' if passed else 'NO_VALIDATED_PHASE_UPDATE'}
        torch.save({'phase':phase.detach().cpu(),'training_indices':chosen},a.output/(name+'.pt'))
        print(name,json.dumps(report[name]),flush=True)
    (a.output/'summary.json').write_text(json.dumps({'identity':'NON_DEPLOYABLE_OPTICAL_OPTIMIZATION_PREFLIGHT','training_rows':len(chosen),
      'official_test_evaluated':False,'conditions':report},indent=2),encoding='utf-8')


if __name__=='__main__': main()
