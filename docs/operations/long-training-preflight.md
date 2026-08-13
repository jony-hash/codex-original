# Long-Training Preflight

This document is the operational companion to the mandatory rule in the root
`AGENTS.md`. It applies to v1, v2, research, diagnostic, one-off, and newly
added training entrypoints.

## Applicability and budget

A run is long when it is expected to use at least 15 GPU-minutes, 30 wall-clock
minutes, or its result will select a model/objective, support a scientific
claim, or authorize a downstream stage.

Use the smallest applicable preflight budget:

| Risk | Preflight cap | Required use |
| --- | ---: | --- |
| Ordinary long run | min(5 minutes, 5% of planned runtime) | representative update |
| Scientific/acceptance run | min(10 minutes, 10% of planned runtime) | update plus counterfactual |

If a representative trial cannot fit the cap, stop and report the compute
blocker. CPU or reduced-model success is not a substitute.

## Required sequence

1. **Static and lineage checks.** Confirm the entrypoint, config, data split,
   initialization, evaluator, device, seed, schedule, and required files. Use
   explicit absolute checkpoint paths; never infer a latest output.
2. **Representative update.** Run three optimizer steps (or the smallest
   batch that exercises every intended trainable group) through the exact
   long-run path. Check finite inputs/loss/gradients/parameters, a real update
   in every intended group, no update in frozen/evaluator parameters, expected
   tensor/physical semantics, and memory headroom.
3. **Counterfactual when direction matters.** For comparison, objective
   selection, or a scientific claim, run the smallest pre-registered negative
   control: for example shuffled labels/codes, a disabled term, a frozen
   control, or mismatched lineage. State the expected rejection first. If it
   passes the claimed gate, classify the harness as invalid and do not train.
4. **Recovery when promised.** Run one interrupted/resumed segment and verify
   the next batch, model, optimizer, RNG, schedule, and history. If recovery
   is not verified, do not claim resumability.

## Evidence and invalidation

Record the exact short-trial command, immutable input paths and hashes, commit,
runtime/device, seed, elapsed time, peak memory, observed gates,
counterfactual result, recovery result, and the one long command authorized by
the trial. A change to training, data selection, loss, optimizer, evaluator,
checkpointing, device, dependencies, or scientific thresholds invalidates the
authorization and requires the affected preflight again. Documentation-only
or unrelated changes do not.

The short trial authorizes only the predeclared long run; it does not predict
final quality. Classify outcomes as:

- `invalid_harness`: environment, lineage, update, semantics, recovery, or
  negative-control failure; repair and rerun the affected preflight.
- `valid_failed`: complete valid measurement missed a scientific gate; stop the
  registered branch without lowering thresholds, extending budget, or retuning
  under the same contract.
- `resource_aborted`: interruption or budget exhaustion; resume only from an
  explicitly verified checkpoint, otherwise stop and report.

Use existing trainer controls and existing evidence boundaries. Do not create
a generic contract framework, public CLI, or evidence schema solely for this
policy.
