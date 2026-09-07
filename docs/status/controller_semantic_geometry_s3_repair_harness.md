# Controller semantic-geometry S3 repair harness

This file records the implementation boundary for the bounded intervention
after `S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED`. It is not an S3 success
record and does not authorize training or formal promotion.

The implementation contract is registered by
`configs/research/lightgen_controller_semantic_geometry_training_v1.yaml` (the
attempt-02 execution config is
`configs/research/lightgen_controller_semantic_geometry_training_v1_attempt_02.yaml`) and is consumed by
`src/ldm_project/v2/application/research/controller_semantic_geometry_training.py`.
The authorized run was performed from clean implementation HEAD
`169fd515b7d6ff6d862222c066c44a39dc373879`.

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

The authorized execution is now complete. Attempt-01 first passed its 3-step
preflight but its training consumer rejected the recorded device alias
`cuda:0` before loading the optical model or creating an optimizer; it wrote
no optimizer update, backward, forward, checkpoint, or S3 result. That
write-once attempt remains preserved at
`outputs/20260907_v2/runs/lightgen_controller_semantic_geometry_s3_repair_v1_attempt_01/`.

## Authorized bounded canary (2026-09-07)

Attempt-02 used the fixed execution config
`configs/research/lightgen_controller_semantic_geometry_training_v1_attempt_02.yaml`
from clean implementation HEAD
`169fd515b7d6ff6d862222c066c44a39dc373879`. The device-alias consumer fix was
`43b1a4250e6b4839d0a32ca0e2ba0ed351335f00`; the attempt-02 config registration
was `169fd515b7d6ff6d862222c066c44a39dc373879`.

The production 3-step CUDA preflight passed with gate
`CONTROLLER_SEMANTIC_GEOMETRY_3_STEP_PREFLIGHT_PASS`. Its write-once evidence is
`outputs/20260907_v2/runs/lightgen_controller_semantic_geometry_s3_repair_v1_attempt_02/preflight.json`
with SHA-256
`ed44ba642252a7ecb08faa2541af9af55f9b7d1a72c9ea8696afd8fc689c1835`; peak CUDA
allocation was `2159626752` bytes against the registered maximum
`5368709120`, and preflight wall time was `52.8974838` seconds. All three
steps had finite nonzero gradients and updates for the four registered
generator parameter groups; frozen-controller negative control, optical/file
identity, evaluator/decoder isolation, and zero preflight checkpoint writes
passed.

The authorized run then completed exactly `492` logical optimizer/backward
steps, consuming the fixed class-balanced R1 geometry batch/code stream. The
only run outputs are `preflight.json` and the final-step checkpoint
`outputs/20260907_v2/runs/lightgen_controller_semantic_geometry_s3_repair_v1_attempt_02/controller_semantic_geometry_final_step_492.pt`,
SHA-256
`4d7c574d98078df38f9866734ad60bacd82c5a7e18dc9d1737fd70e1f42d8d84`. Its
schema is `controller_semantic_geometry_final_controller_state.v1`, step is
`492`, and selection is `FINAL_AUTHORIZED_STEP_ONLY`. The payload records
`loss=0.34952232241630554`, `mean_anchor=0.19811758399009705`, and
`mean_covariance=0.5009270906448364`.

Controller-only isolation passed: starting generator state hash was
`3d06989ba6c5f9154e4d05b64cef42bd4c94b02686ff5b2120a953d9f87a7931`, final
generator state hash was
`cccefcb25c674b11f64653fe1f97a0193d4896bc69c4fe01cf3812daffda1b99`, and
the final checkpoint contains only the four generator state groups. The
formal optical checkpoint, calibration, calibration manifest, evaluator
identity, controller authority, and migrated controller hashes all matched
their frozen bindings. No evaluator, decoder, S3 rerun, generation metric,
or formal promotion was performed.

This is a completed bounded controller training canary, not an S3 efficacy
claim. The classification remains
`S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED`; `formal_promotion=false`, and
the one-shot authorization is not persistent project-wide training authority.
Any scientific effect on the original S3 contract requires a separately
authorized, write-once S3 rerun.

## Post-step-492 frozen S3 consumer (2026-09-07)

The existing post-step-492 candidate was consumed exactly once by the frozen
S3 contract. No retraining, optimizer creation, backward, candidate checkpoint
write, S3 panel change, threshold change, or optical-authority change occurred.
The candidate binding is the final-step-only checkpoint at
`outputs/20260907_v2/runs/lightgen_controller_semantic_geometry_s3_repair_v1_attempt_02/controller_semantic_geometry_final_step_492.pt`,
SHA-256
`4d7c574d98078df38f9866734ad60bacd82c5a7e18dc9d1737fd70e1f42d8d84`; its
preflight SHA is
`ed44ba642252a7ecb08faa2541af9af55f9b7d1a72c9ea8696afd8fc689c1835`, and
its final controller-state hash is
`cccefcb25c674b11f64653fe1f97a0193d4896bc69c4fe01cf3812daffda1b99`.

The first wrapper invocation stopped before S3 artifact creation because the
internal-validation CUDA allocator cache reduced the same-process resource
preflight below the unchanged 4 GiB threshold. The minimal cleanup fix was
committed as `83436fc`; it only releases the internal-validation CUDA objects
and cache. The valid S3 execution then ran from clean HEAD
`83436fcccafb0c760ed81424976fe627a3ef3e2d` and produced the write-once
authority
`outputs/20260907_v2/runs/lightgen_controller_semantic_geometry_s3_post_step_492_v1_attempt_01/`.
Its manifest SHA-256 is
`e13603745ed9fc70ae4f47975241d7653629000090e89aef5cb38b5357e5018c`; all
12 artifact refs and file sizes verify.

The descriptive internal-validation panel used seed `100000` and had zero
overlap with the external S3 panel. The S3 panel stayed seed `1234`,
`Uniform([-1,1]^8)`, 100 base codes reused across 10 labels, 1000 label-major
states, with base-code SHA
`2c83f8fca3d1fde7d5c694f69b602dbc3d43b4e41cfb8159d1af1914efef8c86` and
panel identity SHA
`8a9dd1c7e3e2b674a152e6c96ac8b47976ff1f8d27428a18477e89795b940f47`.

The run was `valid_completed` but failed the scientific writability router:
interface=`S3_CONTROLLER_TO_SEMANTIC_OLS_INTERFACE_PASS`,
targeting=`S3_LABEL_TARGETING_FAIL`, and
reachability=`S3_ABSOLUTE_REACHABILITY_FAIL`. Selection/confirmation real vs
null retrieval was `0.832/0.106` and `0.812/0.098`; targeting direction
consistency was false. Code effect was nondegenerate with no class collapse,
zero exact R1/complex duplicates and 1000 unique R1/complex states. Every
class and the aggregate Jacobian were
`CODE8_LOCAL_CONTROL_FULL_RANK` with effective rank 8. State isolation was
`S3_STATE_ISOLATION_PASS`: candidate, historical controller, optical
checkpoint, calibration, and mode-bank hashes were unchanged; optimizer,
backward, training, checkpoint write, decoder, evaluator, and parameter update
counters were zero.

The final classification is
`S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED`. This closes the current
controller semantic-geometry repair line at the evidence boundary; it does
not authorize S4, another training run, tuning, class-9 repair, or another S3
rerun.
