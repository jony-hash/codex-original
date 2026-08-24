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

## H-DPGC closeout

The retry3 lineage was valid after three earlier device-boundary harness fixes.
Its validation, preflight, audit, report, and closeout hashes are
`af09945e6d085891c6113e19a5330ad480b3ee589e13b99d70fc2a4f1a354ef4`,
`890ad59a26f0c4f3e76099270d38d17da01c60940d4525dbcde8f1d2927a8818`,
`7fc08c89c8d86f31b93873c742496f9e219365980a022b3aea1b0db5eecf501`,
`e3d5737c01886854a65a4d732bc3dba143e13914641313134c10b537bbcbe58b`, and
`e61e40c18a6f09a272a16414ae369699356cd7ca6fc56e666d4f5735b121d075`.

The audit was valid but failed the semantic gate as
`direct_prior_linear_semantic_capacity_failed`. Primary minimum recall on the
three panels was `0.76/0.80/0.92`; secondary minimum recall was
`0.90/0.96/0.96`. The analytic Jacobian rank pass fraction was `1.00` and
exact duplicates were zero, but memorization pass fractions were
`0.914/0.932/0.908`. Official 10k primary/secondary accuracy was
`0.9544/0.9761`, with primary minimum recall `0.8768` and agreement `0.9470`.
Peak allocation was `517880320` bytes and audit storage was `41238` bytes.

This closes only the fixed linear canary. It does not prove that every
nonlinear direct generator is impossible; it does block neural direct-generator
training until a new, independent no-training capacity argument is registered.

## H-SCF closeout: semantic counterfactual safety failure

`H-SCF` (Semantic Code-source Factorial) completed the retry4 manifest-only,
diagnostic-only no-update lifecycle. Validation, CUDA preflight, audit, report,
and closeout are all valid. The final classification is
`semantic_counterfactual_safety_failed`, with the unique next action
`close_frozen_decoder_derived_semantic_route`.

The frozen WAE/H-DIDM decoder × Gaussian/validation-posterior arms were
evaluated with two disjoint 250-per-class posterior banks, two Gaussian banks,
historical panel replay, D-P within-class permutation, label-shift, zero-code,
dual-evaluator, finite, no-update, and anti-collapse controls. Historical view
hashes and evaluator counts, posterior hashes, chunk equality, parameter hashes,
and resource gates passed. Audit results were: S-G safety failed on both banks;
S-P semantic passed on both banks but safety failed; D-G semantic failed on both
banks; D-P semantic failed on both banks and both permutation controls failed.
The safety classification has precedence, so the audit does not claim a unique
decoder-reachability or pairing diagnosis.

Final retry4 evidence hashes are validation
`252b0ebf226859e618c3fdb342a69e80269098bc09d1deebe4b411378d1ab491`, preflight
`25fd79f3b1194299c32f7c0e963849523caf3d4f3cfb596c461bb9a72da05104`, audit
`aca75439aa2fce3f5414f697f34505e91bcc0052c9532aee8ff8b8e24eaef930`, report
`4a60eda420884d7a19d30aa6712482de4f4394aec6cac7e9c8035b3f6c0da9f6`, and
closeout `59d927f7d7ed3b9633fe613e5b9ce14814ba9b5e15776e9ef47352594026ef94`.
Preflight peak allocation was `720658432` bytes, wall time `9.480835378999473`
seconds, and storage `10316` bytes; audit storage was `110601` bytes. No
optimizer, checkpoint, candidate model, confirm, decoder v2, O0-G, optical,
fiber, or promotion consumer is authorized.

## H-NDMC closeout: nonlinear manifold safety failure

H-NDMC completed a valid retry3 manifest-only no-update lifecycle. The canary
was an ephemeral classwise exact PCA-8/Gaussian-copula/continuous-kernel
empirical manifold for the preserved `label + 8D N(0,I) → image` interface; it
did not reuse frozen WAE/H-DIDM decoder parameters and created no optimizer,
checkpoint, candidate model, or deployable software decoder.

The valid classification is `nonlinear_manifold_safety_failed`, with the unique
next action `close_nonlinear_direct_manifold_capacity_canary`. Both Gaussian
banks had primary correct `2500/2500`, secondary correct `2000/2500`, and
agreement `2000/2500`; camera-code fraction was about `1.48e-5`, diversity was
`0.004–0.007`, and coverage was `1.67–2.13`. Memorization and exact-duplicate,
historical replay, finite, chunk, no-update, and resource gates passed.

This valid failure closes the empirical nonlinear canary. It does not authorize
the future architecture preflight, neural training, confirm, decoder v2, O0-G,
optical, fiber, or promotion. The five retry3 evidence hashes are recorded in
the current handoff; outputs and manifests remain read-only and untracked.
