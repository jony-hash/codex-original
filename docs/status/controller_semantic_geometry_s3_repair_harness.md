# Controller semantic-geometry S3 repair harness

This file records the implementation boundary for the bounded intervention
after `S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED`. It is not an S3 success
record and does not authorize training or formal promotion.

The implementation is registered by
`configs/research/lightgen_controller_semantic_geometry_training_v1.yaml` and
is consumed by
`src/ldm_project/v2/application/research/controller_semantic_geometry_training.py`.

## Frozen consumer contract

- The real target is the immutable S1 `R1_POWER` fit bank, partitioned into
  500 training states and 500 internal-validation states, 50 per class.
- The target-bank artifact contains only `training` and
  `internal_validation` tensor groups. Its manifest binds the explicit
  partition-manifest SHA, indices, labels, R1 tensor hashes, observer, and
  frozen optical lineage. The sidecar manifest must bind the serialized
  target-bank SHA before a consumer loads it. Runtime resplitting and
  reselection are rejected.
- The S3 seed-1234 base-code panel remains external scientific evaluation.
  The internal-validation code panel is the separate seed-100000 panel.
  Training codes use one CPU `torch.Generator`, the registered derived seed,
  one sequential `[50, 8]` draw per logical step, and exact overlap checks.
- The only objective is
  `CLASS_CONDITIONED_R1_FIRST_SECOND_ORDER_GEOMETRY_ALIGNMENT`: float32
  class means and `N-1` centered covariance, real-only positive
  normalizers, equal 0.5/0.5 mechanism aggregation, and equal class mean.
  No decoder, relay, CMOS, evaluator, or auxiliary loss enters the graph.
- The optimizer is one fresh controller-only Adam instance with the four
  registered generator parameters and fixed `2e-4` learning rate. Optical,
  OLS, calibration, relay, detector, and evaluator state are frozen and
  hashed before and after an authorized run.
- The horizon is exactly 492 logical steps. Only the exact post-step-492
  controller state may be written; validation and S3 cannot select a
  checkpoint.

## Execution boundary

The no-update preflight may construct isolated CPU fixtures, build a fresh
optimizer, evaluate target moments, materialize the finite CPU code stream,
and verify hashes. It performs zero optimizer steps, zero backward calls, and
does not mutate a registered controller. The training CLI rejects every
authorization value other than:

`AUTHORIZE_ONE_SHOT_CONTROLLER_SEMANTIC_GEOMETRY_TRAINING_CANARY`

No training, 492-step canary, or S3 rerun is part of this implementation
commit. Scientific success remains undecided until the original frozen S3
contract is rerun under a separately authorized write-once attempt.
