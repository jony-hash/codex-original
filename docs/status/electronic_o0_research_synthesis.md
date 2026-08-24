# Electronic O0 research synthesis (2026-08-24)

This document is the cross-route decision record for the electronic O0
capacity-map program. It is a governance artifact: it consumes only complete
valid closeouts and does not create a model, select a checkpoint, or authorize
training.

## Comparable evidence

All rows below use the WAE v2 seed-7301 source, the registered MNIST split, and
the qualified primary/secondary evaluators unless stated otherwise. Metrics are
kept in separate axes; no aggregate score is formed from incomparable losses.

| route | semantic axis | prior/distribution axis | cycle/reconstruction axis | safety/governance decision |
| --- | --- | --- | --- | --- |
| corrected WAE | primary `0.930–0.952`, secondary `0.826–0.852`, secondary min recall `0.36–0.44` | standard Gaussian decoder input, no prior-native fit gate | reconstruction passed its local contract; image-manifold gate failed | close WAE image-manifold route |
| latent flow | locked minimum-recall and memorization gates failed | posterior transport improved only locally | no downstream acceptance | close latent transport |
| class mixture | no evaluator confirmation authorized | SWD reduction `12.58%`, novelty pass `0.894` | no downstream acceptance | close mixture sampler |
| H-SWAE | aggregate SWD reduction `34.62%`; classes 6/7 regressed | prior-native fit below `50%` gate | cycle worsened | close route before confirm |
| H-SCED | training/reference SWD reductions `39.73%/36.67%` | aggregate SCED `0.03904→0.02047` | reconstruction improved, cycle `0.01649→0.02251` | close route before confirm |
| H-PGC/TCA/DLTA | diagnostic only | no stable gradient, optimizer, curvature, or observer attribution | no supported mechanism | close optimizer/trajectory attribution family |
| H-CCF | diagnostic only | not a prior-fit test | encoder/decoder/interaction effects cancel; composition underdetermined | close prior-native attribution family |

The WAE, SWAE, and SCED numbers are not interchangeable acceptance metrics:
the former is an image/evaluator test, while the latter two are prior-alignment
and cycle diagnostics. The attribution routes contain no candidate model and
therefore cannot be ranked on the semantic axes.

## Decision

The following decisions are now fixed:

- `prior_native_attribution_family_closed=true`.
- No existing checkpoint has promotion, confirm, optical, fiber, or O0-G
  permission.
- Encoder and cycle were training mechanisms in the closed WAE-derived routes;
  they are not required by the final single-pass inference interface.
- The preserved interface for any future electronic hypothesis is
  `label + 8D N(0,I) → image`, with single-pass/single-exposure inference.
- No further epoch extension, optimizer swap, weight retuning, seed rotation,
  or attribution replay is authorized for the closed routes.

## Next hypothesis and consumer

`H-DPGC` (Direct Prior Generator Capacity Canary) is registered as a
diagnostic-only, no-optimizer capacity test. It constructs a fixed, ephemeral
classwise rank-8 logit-Gaussian canary from the training partition and measures
whether the preserved interface has enough direct capacity evidence. It does
not claim that a linear canary bounds all nonlinear generators.

The canary's only valid consumer is its own report/closeout and this ledger:

- a pass may register a future direct-generator architecture preflight;
- a valid failure closes this linear canary and requires new no-training
  capacity evidence;
- neither result authorizes neural training or formal promotion.

All old checkpoints, manifests, evidence, and outputs remain read-only.
