"""Fit a small digital linear probe on frozen C3-C CMOS diagnostic images."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.research.lightgen.hw207_mnist4_c31_c32_run_v1 import load_inputs
from scripts.research.lightgen.hw207_mnist4_c31_c32_cmos_v1 import prepare_configuration, forward_hw207_mnist4
from scripts.research.lightgen.hw207_mnist4_c0_v1 import metrics


def extract(x, indices, context, batch):
    r0, r1 = context.rois[0][0], context.rois[3][1]
    c0, c1 = context.rois[0][2], context.rois[3][3]
    result = []
    for ids in indices.split(batch):
        pixels = forward_hw207_mnist4(x[ids], context)['cmos_pixel_integral']
        result.append(F.adaptive_avg_pool2d(pixels[:, None, r0:r1, c0:c1].float(), (28, 28)).flatten(1).cpu())
    return torch.cat(result)


def fit(train, labels, steps=None):
    torch.manual_seed(11)
    model = torch.nn.Linear(784, 4).to(train.device)
    initial = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=.15, weight_decay=.001)
    rng = torch.Generator().manual_seed(11)
    count = 0
    for _ in range(30):
        for ids in torch.randperm(len(labels), generator=rng).split(256):
            ids = ids.to(train.device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(train[ids]), labels[ids])
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite probe loss or gradient')
            optimizer.step(); count += 1
            if count == steps:
                if any(torch.equal(a, b) for a, b in zip(initial, model.parameters(), strict=True)):
                    raise ValueError('Probe weight or bias did not update')
                return model
    return model


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--indices',type=Path,required=True)
    p.add_argument('--train-pt',type=Path,required=True); p.add_argument('--candidate-config',type=Path,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--batch',type=int,default=8); a=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    x,y,val=load_inputs(root,a.train_pt,a.indices)
    split=json.loads(a.indices.read_text(encoding='utf-8')); train=torch.tensor(split['train']['original_indices'])
    rng=torch.Generator().manual_seed(207032)
    train=torch.cat([train[torch.where(y[train]==k)[0][torch.randperm(int((y[train]==k).sum()),generator=rng)[:256]]] for k in range(4)])
    candidate=json.loads(a.candidate_config.read_text(encoding='utf-8'))['relay_candidates'][2]
    context=prepare_configuration(candidate,a.device)
    a.output.mkdir(parents=True,exist_ok=False)
    tr=extract(x,train,context,a.batch); va=extract(x,val,context,a.batch)
    scale=tr.square().sum(1).mean().sqrt().item()
    if not scale>0: raise ValueError('Zero training feature scale')
    tr=(tr/scale).to(a.device); va=(va/scale).to(a.device); ty=y[train].to(a.device)
    fit(tr,ty,steps=3)
    shuffled=ty[torch.randperm(len(ty),generator=torch.Generator().manual_seed(207033)).to(a.device)]
    negative=fit(tr,shuffled)
    neg=metrics(negative(va).detach().cpu(),y[val])
    if neg['accuracy'] >= .40: raise ValueError('Random-label negative control did not reject')
    model=fit(tr,ty); result=metrics(model(va).detach().cpu(),y[val])
    torch.save({'train_indices':train,'validation_indices':val,'train_features':tr.cpu(),'validation_features':va.cpu(),
                'weight':model.weight.detach().cpu(),'bias':model.bias.detach().cpu(),'scale':scale},a.output/'probe.pt')
    report={'status':'AUXILIARY_DIGITAL_LINEAR_PROBE','candidate':'C3-C','training_rows':len(train),'validation_rows':len(val),
            'feature':'projected CMOS rectangle, 28x28 adaptive average pool; train-only global RMS scale',
            'preflight':'three finite SGD steps with weight and bias updates','random_label_validation':neg,
            'validation':result,'official_test_evaluated':False,'optical_parameters_updated':False}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report),flush=True)


if __name__=='__main__': main()
