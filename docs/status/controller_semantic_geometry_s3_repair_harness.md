# Controller semantic-geometry S3 repair harness

This file records the implementation boundary for the bounded intervention
after `S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED`. It is not an S3 success
record and does not authorize training or formal promotion.

The implementation is registered by
`configs/research/lightgen_controller_semantic_geometry_training_v1.yaml` and
is consumed by
`src/ldm_project/v2/application/research/controller_semantic_geometry_training.py`.

## Materialized data authority (2026-09-07)

The previous authorized execution at `e4e1a855f55a542d75ec5121b86dd242028c71b6` stopped as
`INVALID_HARNESS` at step 0. The dedicated write-once authority is
`outputs/20260907_v2/runs/lightgen_controller_semantic_geometry_data_firewall_v1_attempt_01/`.
Its target bank SHA is
`d0f29701e4f9818c48b222e1cbfd8706257ec38356842870f154cd3fc2c8ecb0`; its partition manifest SHA is
`1eae38d6086a7e395960177e24e8b1ccb7deb1312f2885c07321f3ebeb1cd7c4`; its target-bank sidecar SHA is
`9eb2f032de52ec68dfda69ad3abd217ded2774656ec83c217ebe7d22ebed7916`; and its data-firewall manifest
SHA is `f5704a5108681a78273b918d3eda3ade3b0e68b8af6f4e98314037e1eb94d4d1`.

The frozen semantic partition identity is
`b07a04d3c779e5952c06209c9d8cc8421b8193cd41f037db88f95c736ab35594`; it is deliberately distinct from
the physical partition-manifest SHA. The S1 fit source and deterministic label-major/source-index rule
reproduce all frozen row-position, source-index, label, and R1 component hashes. The registrar records
the independently computed canonical material digest
`2010a7541994c9422de26f5f646ae67e6c91f73ec49297a2a4f378589615c431`. No older repository authority
contains a canonical-byte recipe mapping that material digest to `b07…`; consequently the consumer treats
`b07…` as the frozen semantic authority and never equates it with the physical file SHA or the local material
digest.

The target bank is redacted to `training` and `internal_validation` only, reloads through the production
safe loader, and has zero overlap with S1 selection, S1 confirmation, S2 validation, S3 base codes, or the
internal-validation code panel. Registration classification is
`CONTROLLER_SEMANTIC_GEOMETRY_DATA_AUTHORITY_REGISTERED`, with zero optimizer, backward, and forward calls.
On clean implementation HEAD `0f1126660a2d2d8c497c7c35d3ba01fd5c6d13d8`, the production no-update preflight
returned `PRETRAINING_HARNESS_PREFLIGHT_PASS` with zero optimizer steps, zero backward calls, and zero
checkpoint writes. The implementation status is
`IMPLEMENTATION_READY_FOR_ONE_SHOT_REAUTHORIZATION`; this does not authorize training or S3 rerun.

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
