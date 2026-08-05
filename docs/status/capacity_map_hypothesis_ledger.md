# Capacity-map hypothesis ledger

This ledger is the single mutable index for the diagnostic-only capacity-map
program. Evidence files are immutable; a repaired harness receives an evidence
revision and never a silent architecture-version bump. Rows that describe
O1--O5 are historical preregistration context only: O0 failed and no physical
capacity attribution is currently permitted.

| ID | Unique variable | Prediction | Falsification / attribution rule | Status |
| --- | --- | --- | --- | --- |
| H-E | E0/E1 versus E2--E4 seed manifold | A spatial nonlinear seed raises a Pareto axis without reducing the other two | E2--E4 pass while E0/E1 fail: input-manifold bottleneck | v2.2 full-panel replay of revision-9 E2:O0: primary min=0.876, smooth min=0.488, agreement min=0.482, min recall=0; no optical attribution |
| H-R | O0 versus O1 | Digital oracle pass and relay failure isolates relay/detector representation | O0 pass, O1 fail: relay/representation bottleneck | preregistered |
| H-P | O1 versus O2 | Phase-only 2+2 reduces an otherwise valid response | O1 pass, O2 fail: D2NN/crop/phase-only transfer | pending: O1 now uses diagnostic bypass; all physical probes screened out |
| H-L | O2 versus O3 | 4+4 near-conjugate mixing unlocks a blocked Pareto point | O3 pass, O2 fail: layer/mode-mixing bottleneck | provisional single-seed: primary 0.50 vs 0.20; smooth remains 0.10 |
| H-S | O2/O3 versus O4 | More physical spatial degrees of freedom unlock the response | O4 pass only: spatial-DOF bottleneck | provisional negative: O4 matched O2 at 0.20 primary |
| H-C | O5 versus phase-only | Complex control unlocks response absent under phase-only control | O5 pass, phase-only fails: phase-only capacity bottleneck | provisional negative after identity-init repair: O5 matched O2 |
| H-T | Teacher/GAN upper bounds | A learned upper bound separates optimization from physical reachability | upper bound passes while teacher-free fails: objective/optimization, not physics | preregistered |
| H-M | Fixed global exposure | A single global exposure rescues both qualified evaluators without saturation | Any selected exposure must be real-data qualified and shared by both evaluators | v2.2 holdout sweep: no scale jointly improves smooth/agreement while retaining saturation ≤0.08; report-only, no selection |
| H-Q | Third Shape evaluator | A spatially distinct evaluator can adjudicate the primary/smooth disagreement | It may enter diagnostic agreement only after v2.1-equivalent qualification | valid local failure: specified global-average architecture, seed 7319, 10 epochs reached 0.9234 validation accuracy; excluded from adjudication |
| H-O | Dual-bundle objective | Normalized primary+smooth loss improves smooth semantics without primary regression | 2/3 seeds: smooth median gain ≥0.10, primary loss ≤0.02, no safety failure | smoke selection passed (smooth median 0.571 vs 0.100); 5-epoch three-seed E2:O0 confirm failed all complete gates: smooth 0.747/0.753/0.747, agreement 0.735/0.741/0.671, min recall 0.412/0.353/0.294 |

Research failures close only their own variant. Formal acceptance remains
fail-closed. Invalid harness records are excluded from Pareto ranking and may
not support a capacity conclusion.

## Closed v2.2 branch

Status: `closed_conditional_negative`. The immutable closeout is archived in
[`experiments/archive/v2_2_capacity_map_closeout/`](../../experiments/archive/v2_2_capacity_map_closeout/),
including its summary, manifest, report, and source evidence index. The
original `outputs/` run remains a source location, not the only reference.

Allowed claim: the current `E2:O0_declared_teacher_free_objectives` family did
not pass the qualified dual-evaluator gate (`confirmation_pass_count=0/3`).
The causal label remains `training_objective_conflict`, with certainty
`not_uniquely_identified`; the retained alternatives are electronic generator
capacity, optimization dynamics, and evaluator disagreement.

Prohibited claims: optical-capacity failure, phase-only-capacity failure,
spatial-DOF failure, and a universal semantic--diversity capacity conflict.
O1--O5, Teacher/WGAN, Controlled Shapes, and Fashion-MNIST are unexecuted
because the O0 gate stopped downstream work; no formal promotion is authorized.

## Boundary classes fixed at closeout

- Established: all three E2:O0 dual confirms failed the complete qualified
  dual-evaluator gate.
- Not uniquely identified: electronic generator capacity, optimization
  dynamics, and evaluator disagreement.
- Untested: optical capacity, phase-only control, spatial DOF, and cross-dataset
  generality. A future investigation must begin with a new electronic O0
  contract and objective family, not a continuation of O1--O5.

## v2.3 electronic-oracle preregistration

The branch `experiment/v2-o0-feasibility-v23` is a new, research-only contract
based on the immutable v2.2 closeout. Its sole question is whether a teacher-free
single-forward electronic amplitude-command generator can pass the qualified
dual-evaluator protocol before any optical work is reconsidered. The ordered
counterfactuals are D0 (28x28 image upper bound), C0 (current 8D exact anchor),
C1 (higher-capacity 8D exact anchor), C2 (higher-capacity 8D soft anchor), and
C3 (32D soft diagnostic upper bound). D0 failure stops the branch; C3 can never
authorize physical follow-up.

Implementation, revision-2 controls and CUDA preflight passed. The preregistered
D0 seed-1234 probe then failed validly at step 50: worst normalized margin did
not improve and digital code fraction fell from approximately 0.0323 to 0.0020;
finite, safety and saturation checks remained valid. The branch therefore closes
as `training_objective_or_evaluator_inconclusive` for the D0 image-domain
teacher-free objective only. Smoke, confirm, locked-final, command candidates,
and all optical branches were stopped and must not be inferred from this result.
