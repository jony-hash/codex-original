# Long-Training Preflight

This document is the operational companion to the mandatory rule in the root
`AGENTS.md`. It applies to v1, v2, research, diagnostic, one-off, and newly
added training entrypoints.

## Applicability and startup checks

A run is long when it is expected to use at least 15 GPU-minutes, 30 wall-clock
minutes, or its result will select a model/objective, support a scientific
claim, or authorize a downstream stage.

There is no elapsed-time cap for preflight or scientific execution. Use the
smallest representative check, record its duration, and recheck only behavior
changed by a repair. Estimates inform scheduling and do not reject execution.
Approved update counts and scientific populations remain fixed. CPU or
reduced-model success is not a substitute for required CUDA verification.

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

Record the exact short-trial command, explicit input paths and loaded identities, commit,
runtime/device, seed, elapsed time, peak memory, observed gates,
counterfactual result, recovery result, and the one long command authorized by
the trial. A change to training, data selection, loss, optimizer, evaluator,
checkpointing, device, dependencies, or scientific thresholds requires checking
the affected production behavior again. Ordinary repairs remain under the same
task authorization. Do not use hash or source-manifest equality as an execution
lock. Documentation-only or unrelated changes do not require another preflight.

The short trial authorizes only the predeclared long run; it does not predict
final quality. Classify outcomes as:

- `invalid_harness`: environment, lineage, update, semantics, recovery, or
  negative-control failure; repair and rerun the affected preflight.
- `valid_failed`: complete valid measurement missed a scientific gate; stop the
  registered branch without lowering thresholds, increasing update counts, or retuning
  under the same contract.
- `resource_aborted`: interruption or an actual hardware resource failure; resume only from an
  explicitly verified checkpoint, otherwise stop and report.

Use existing trainer controls and existing evidence boundaries. Do not create
a generic contract framework, public CLI, or evidence schema solely for this
policy.
